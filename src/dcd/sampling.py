import abc
import numpy as np
import torch
import torch.nn.functional as F
import math
from src.sedd.catsample import sample_categorical

from src.sedd.model import utils as mutils
from tqdm import tqdm

import time
import re


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, autoregressive_lm, x, t, step_size, **kwargs):
        """One update of the predictor.

        Args:
            score_fn: score function
            autoregressive_lm: GPT model
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


def scaled_autoregressive_generation(model, x, ref_probs, target_probs, alpha = 1.0):

    B = ref_probs.size(0)

    log_odds = target_probs.clip(min = 1e-16).log() - ref_probs.clip(min = 1e-16).log()

    # Initialize the generated sequence with the input_ids
    generated_ids = torch.multinomial(target_probs[:,0,:], num_samples = 1)
    mask = (x[:,0] != 50257)
    generated_ids[mask,0] = x[mask,0]

    # Initialize past_key_values to None
    past_key_values = None

    # Iterate to generate tokens one by one
    for i in range(1, target_probs.size(1)):
        # Get model outputs (only the last token's logits)
        with torch.no_grad():
            outputs = model(input_ids = generated_ids[:, -1:], past_key_values = past_key_values, use_cache = True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values  # Update the kv cache
        
        # Get the logits for the last generated token
        next_token_logits = logits[:, -1, :]
        rescaled_logits = next_token_logits + log_odds[:,i,:] * alpha
        
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(rescaled_logits, dim = -1)

        # Sample the next token from the probability distribution
        next_token_id = torch.multinomial(probabilities, num_samples = 1).reshape(B, 1)

        # Overwrite by previously generated tokens
        mask = (x[:,i] != 50257)
        next_token_id[mask,0] = x[mask,i]
        
        # Append the sampled token to the generated sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim = -1)

    return generated_ids


def scaled_partial_autoregressive_generation(model, x, ref_probs, target_probs, decode_sid, decode_eid, kv_cache = None):

    x = x.detach().clone()

    B = ref_probs.size(0)

    log_odds = target_probs.clip(min = 1e-16).log() - ref_probs.clip(min = 1e-16).log()

    generated_ids = torch.zeros([B, 0], dtype = torch.long, device = x.device)
    past_key_values = kv_cache

    sid, eid = decode_sid.amin().item(), decode_eid.amax().item()
    for token_id in range(sid, eid):

        if token_id == 0:
            generated_ids = torch.multinomial(target_probs[:,0,:], num_samples = 1)
        else:
            # Get model outputs (only the last token's logits)
            with torch.no_grad():
                input_ids = x[:,sid-1:sid] if token_id == sid else generated_ids[:, -1:]
                input_ids = input_ids.contiguous()
                outputs = model(input_ids = input_ids, past_key_values = past_key_values, use_cache = True)
                logits = outputs.logits
                past_key_values = outputs.past_key_values  # Update the kv cache

            # Get the logits for the last generated token
            next_token_logits = logits[:, -1, :]
            rescaled_logits = next_token_logits + log_odds[:,token_id-sid,:]
            
            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(rescaled_logits, dim = -1)

            # Sample the next token from the probability distribution
            next_token_id = torch.multinomial(probabilities, num_samples = 1).reshape(B, 1)

            # Overwrite by previously generated tokens
            mask = (x[:,token_id] != 50257) | ((token_id < decode_sid) | (token_id >= decode_eid))
            next_token_id[mask,0] = x[mask,token_id]
            
            # Append the sampled token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim = -1)

    tids = torch.arange(sid, eid, device = x.device)
    mask = (tids[None,:] >= decode_sid[:,None]) & (tids[None,:] < decode_eid[:,None])
    mask_i, mask_j = torch.where(mask)
    x[mask_i, mask_j + sid] = generated_ids[mask_i, mask_j]

    return x, past_key_values


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    """
    The original sampler from SEDD.
    """
    def update_fn(self, nc_score_fn, c_score_fn, autoreg_lm, x, t, step_size, init_flag = False):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        final_sigma = self.noise(torch.zeros_like(t))[0]
        dsigma = curr_sigma - next_sigma

        score = nc_score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        probs = probs.clip(min = 0.0)

        absorb_prob = probs[:,:,-1] / probs[:,:,:].sum(dim = 2)

        x_rec = sample_categorical(probs[:,:,:-1])
        absorb_mask = torch.rand_like(absorb_prob) < absorb_prob
        x_rec[absorb_mask] = 50257

        return x_rec


@register_predictor(name="autoregressive")
class AutoregressivePredictor(Predictor):
    def update_fn(self, nc_score_fn, c_score_fn, autoreg_lm, x, t, step_size, init_flag = False):
        
        c_probs = torch.ones([x.size(0), x.size(1), 50257], dtype = torch.float32, device = x.device)
        nc_probs = torch.ones([x.size(0), x.size(1), 50257], dtype = torch.float32, device = x.device)
        
        x_rec = scaled_autoregressive_generation(autoreg_lm, x, c_probs, nc_probs)

        return x_rec


@register_predictor(name="DCD_chunked")
class CombinedCopulaPredictor(Predictor):
    """
    Discrete Copula Diffusion with chunked predictions.
    """
    def update_fn(self, nc_score_fn, c_score_fn, autoreg_lm, x, t, step_size, init_flag = False, alpha = 1.0):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        nc_score = nc_score_fn(x, curr_sigma)
        c_score = c_score_fn(x, curr_sigma)

        nc_stag_score = self.graph.staggered_score(nc_score, dsigma)
        nc_probs = nc_stag_score * self.graph.transp_transition(x, dsigma)
        nc_probs = nc_probs.clip(min = 0.0)

        c_stag_score = self.graph.staggered_score(c_score, dsigma)
        c_probs = c_stag_score * self.graph.transp_transition(x, dsigma)
        c_probs = c_probs.clip(min = 0.0)

        absorb_prob = nc_probs[:,:,-1] / nc_probs[:,:,:].sum(dim = 2)

        nc_probs = nc_probs[:,:,:-1]
        nc_probs /= nc_probs.sum(dim = 2, keepdim = True)

        c_probs = c_probs[:,:,:-1]
        c_probs /= c_probs.sum(dim = 2, keepdim = True)

        x_rec = scaled_autoregressive_generation(autoreg_lm, x, c_probs, nc_probs, alpha = alpha)

        if (t - step_size > 0.1).all():
            chunk_size = ((curr_missing_frac - next_missing_frac) * seq_len).round().long().clip(max = seq_len).max().item()
            chunk_size = max(chunk_size, 8)
            absorb_mask = torch.rand_like(absorb_prob[:,::chunk_size]) < absorb_prob[:,::chunk_size]
            absorb_mask = absorb_mask[:,:,None].repeat(1, 1, chunk_size).flatten(1, 2)
            x_rec[absorb_mask] = 50257

        return x_rec


@register_predictor(name="DCD_autoregressive")
class CombinedAutoregressiveCopulaPredictor(Predictor):
    """
    Discrete Copula Diffusion with autoregressive predictions.
    """
    def __init__(self, graph, noise):
        super().__init__(graph, noise)

        self.kv_cache = None
        self.decode_sid = None

    def update_fn(self, nc_score_fn, c_score_fn, autoreg_lm, x, t, step_size, init_flag = False):
        seq_len = x.size(1)

        if init_flag:
            self.decode_sid = torch.zeros([x.size(0)], dtype = torch.long, device = x.device)

        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        next_missing_frac = self.graph.get_missing_probs_from_sigma(next_sigma)
        decode_eid = ((1.0 - next_missing_frac.squeeze(1)) * seq_len).round().long().clip(max = seq_len)

        sid, eid = self.decode_sid.amin().item(), decode_eid.amax().item()

        nc_score = nc_score_fn(x, curr_sigma)[:,sid:eid,:]
        c_score = c_score_fn(x, curr_sigma)[:,sid:eid,:]

        nc_stag_score = self.graph.staggered_score(nc_score, dsigma)
        nc_probs = nc_stag_score * self.graph.transp_transition(x[:,sid:eid], dsigma)
        nc_probs = nc_probs.clip(min = 0.0)
        nc_probs = nc_probs[:,:,:-1]
        nc_probs /= nc_probs.sum(dim = 2, keepdim = True)

        c_stag_score = self.graph.staggered_score(c_score, dsigma)
        c_probs = c_stag_score * self.graph.transp_transition(x[:,sid:eid], dsigma)
        c_probs = c_probs.clip(min = 0.0)
        c_probs = c_probs[:,:,:-1]
        c_probs /= c_probs.sum(dim = 2, keepdim = True)

        x_rec, kv_cache = scaled_partial_autoregressive_generation(
            autoreg_lm, 
            x, 
            ref_probs = c_probs, 
            target_probs = nc_probs, 
            decode_sid = self.decode_sid, 
            decode_eid = decode_eid,
            kv_cache = self.kv_cache
        )
        self.kv_cache = kv_cache

        self.decode_sid[:] = decode_eid

        return x_rec

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t, **kwargs):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)


def get_combined_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    try:
        predictor = get_predictor(predictor)(graph, noise, num_steps = steps)
    except Exception:
        predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def combined_sampler(diffusion_lm, autoregressive_lm):
        sampling_noncausal_score_fn = mutils.get_score_fn(diffusion_lm, train=False, sampling=True, causal=False)
        sampling_causal_score_fn = mutils.get_score_fn(diffusion_lm, train=False, sampling=True, causal=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        try:
            predictor.kv_cache = None
        except Exception:
            pass

        for i in tqdm(range(steps), disable = True):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_noncausal_score_fn, sampling_causal_score_fn, autoregressive_lm, x, t, dt, init_flag = (i == 0))
            
        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_noncausal_score_fn, x, t)
            
        return x
    
    return combined_sampler