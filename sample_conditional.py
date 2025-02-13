import torch
import argparse
import re
import math
import os
import sys

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import src.dcd.sampling as sampling
from src.sedd.load_model import load_model


def parse_ranges(s, seq_len):
    # Regular expression to find patterns like [0.25-0.75]
    matches = re.findall(r'\[(\d+\.\d+)-(\d+\.\d+)\]', s)
    
    # Convert matches to a list of tuples
    tuples_list = [(int(math.floor(float(start) * seq_len)), int(math.floor(float(end) * seq_len))) for start, end in matches]
    
    return tuples_list


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional generation with Discrete Copula Diffusion")

    parser.add_argument("--diffusion-model", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--gpt-model", default="gpt2", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--sample-type", type=str, default="DCD_autoregressive")
    parser.add_argument("--prompt-template", type=str, default="[0.1-0.2][0.5-0.7]")

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    device = torch.device('cuda:0')

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Diffusion model
    model, graph, noise = load_model(args.diffusion_model, device)
    
    # GPT model
    autoreg_lm = GPT2LMHeadModel.from_pretrained(args.gpt_model)
    autoreg_lm.to(device)

    # Mask
    mask_ranges = parse_ranges(args.prompt_template, args.seq_len)

    def apply_mask_fn(samples):
        input_ids = []
        input_locs = []
        for sid, eid in mask_ranges:
            input_ids.append(samples[:,sid:eid])
            input_locs.extend(list(range(sid, eid)))

        input_ids = torch.cat(input_ids, dim = 1).detach().clone().to(device)
        input_locs = torch.tensor(input_locs).detach().clone().to(device)

        def proj_fn(x):
            x[:,input_locs] = input_ids.clone()
            return x

        return proj_fn

    # Example sentence
    sentence = "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . " + \
               "While it retained the standard features of the series , it also underwent multiple adjustments , such as making the " + \
               "game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned " + \
               "from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . " + \
               "The game 's opening theme was sung by May 'n ."

    # Tokenize
    samples = torch.tensor(tokenizer.encode(sentence)[:args.seq_len])[None,:]

    # Generate project function
    proj_fun = apply_mask_fn(samples)

    # Get the sampler
    sampling_fn = sampling.get_combined_sampler(
        graph, noise, (args.batch_size, args.seq_len), args.sample_type, args.steps, device=device, proj_fun=proj_fun
    )

    # Sample
    pred = proj_fun(sampling_fn(model, autoreg_lm))

    # Detokenize
    pred_samples = tokenizer.batch_decode(pred)
    print(pred_samples[0])


if __name__ == "__main__":
    main()
