import torch
import argparse
import re
import math
import os
import sys

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import src.dcd.sampling as sampling
from src.sedd.load_model import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional generation with Discrete Copula Diffusion")

    parser.add_argument("--diffusion-model", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--gpt-model", default="gpt2", type=str)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--sample-type", type=str, default="DCD_autoregressive")

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

    # Get the sampler
    sampling_fn = sampling.get_combined_sampler(
        graph, noise, (1, args.seq_len), args.sample_type, args.steps, device=device
    )

    # Sample
    pred = sampling_fn(model, autoreg_lm)

    # Detokenize
    pred_samples = tokenizer.batch_decode(pred)
    print(pred_samples[0])


if __name__ == "__main__":
    main()
