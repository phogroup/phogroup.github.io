---
layout: page
title: Autoencoder Compression to Accelerate Inverse Problems
permalink: /nsfcareer/year2/compression/
---

### Major Activities 
Time-dependent PDE constrained inverse problems
are notorious not only for their difficulty in solving, but also for their large memory requirement
--- making it intractable to naively use state-of-the-art
adjoint based techniques. Current approaches to mitigate the intractable memory requirements
use a checkpointing strategy whereby only a small fraction of the state solutions generated
by solving the forward PDE are stored. Checkpointing strategies trade storage for computation
by requiring duplicate solves of the forward PDE during the adjoint phase to recover the
discarded state information. Our approach compresses the state during the forward phase
using the encoding portion of
a pre-trained autoencoder in order to be able to store all of the states in memory ---
eliminating the need for checkpointing and additional PDE solves. The states are then
decompressed as needed during the adjoint phase using the decoder portion of the autoencoder.


### Significant Results

We have demonstrated that the autoencoder compression algorithm works for states generated by a variety of PDEs including 2D Laplace equation, 2D and 3D acoustic wave equation, and 3D acoustic-elastic wave equation. Shown below is a video comparing the original 3D state of an acoustic-elastic wave equation with the decompressed state using a compression ratio of 64:1. The mean relative error between the two states is 5% over the entire testing dataset. 


[![acoustic-elastic wave equation video](/assets/figures/jon/mangll_animation_frame.png)](/assets/figures/jon/mangll_animation_trimmed.ogv "Mangll video")
