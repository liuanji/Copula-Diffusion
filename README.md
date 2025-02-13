# Discrete Copula Diffusion

The official implementation of "Discrete Copula Diffusion" published at ICLR 2025.

## What Is This Paper About?

Diffusion models for discrete data excel at modeling text, but they need hundreds to thousands of diffusion steps to perform well.

We show that this is caused by the fact that discrete diffusion models predict each output token **independently** at each denoising step.

![Alt Text](figs/DCD.png)