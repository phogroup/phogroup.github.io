---
layout: page
title: Uncertainty Quantification for Deep Learning 
permalink: /nsfOAC/UQ/
---
### Uncertainty Quantification for Deep Learning

Model-constrained deep neural networks are capable of learning from both training data and the mathematical models governing a phenomenon of interest; these methods have direct applicability to inverse problems. In this project we extend model-constrained deep neural networks to a model-constrained Bayesian neural network (MCBNN) form, facilitating the solution of inverse problems while simultaneously quantifying uncertainty in a physically interpretable manner. We produce MCBNN forms for a single network learning the inverse map, as well as autoencoders learning both the forward and inverse maps simultaneously. In addition, we have a form of MCBNN which learns a Tikhonov inverse solver with quantified uncertainty. Furthermore, novel training methods for MCBNN allow us to sidestep the common Bayesian issue of defining a prior for the Bayesian neural network parameters by eliciting a physical prior in the network parameter space.

### Results

We are testing the performance of MCBNN by solving three inverse problems governed by Poisson's equation, Burger's equation, and the Navier-Stokes equations', respectively. The below figure demonstrates basic qualitative results for one sample when solving the Poisson inversion, illustrating the predictive accuracy of MCBNN, as well as the uncertainty quantification capability of MCBNN.

![PDE parameters solved when inverting the Poisson equation. From left to right; true solution, predicted solution, absolute error of prediction, standard deviation of prediction. Observation locations indicated by dots.](/assets/figures/rusty/mcbnn_poisson_no_titles.png "fig:mcbnn_heateq")

Furthermore, tests have shown that training MCBNN with a "warm-start" --- a way to sample particles from the physical prior --- enables the network to learn faster than networks with initial parameters sampled from a simple, non-physical distribution like a Gaussian.