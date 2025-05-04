---
layout: page
title: Learning Navier Stokes in the Frequency Domain
permalink: /nsfOAC/fourier-learning/
---
### Motivation
Though initial results for learning approximate solutions to Navier Stokes problems using the [mcTangent](https://arxiv.org/abs/2208.04995) method have been promising, making accurate predictions at time steps well past the end of the training dataset remains difficult. A new strategy, based on [latent space learning](https://arxiv.org/abs/2301.10391) may help to combat this problem. By converting the dataset into the frequency domain and using neural networks to learn the evolution of the Fourier coefficients through time, we hope to learn a more accurate approximate solution at later timesteps.

![Methodology](/assets/figures/cole/fourier_mc_tangent.png)
