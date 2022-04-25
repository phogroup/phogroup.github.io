---
layout: page
title: CDSE Project
permalink: /cdse/
---

# Major Goals of the Project

Our objective in this proposal is to develop an integrated research program that addresses data management and data analytics challenges arising from observations and simulations in UQ and complex inverse problems.

## Aim 1: Developing scalable algorithms for managing large simulation data.
One of the main bottlenecks to data scalability of existing inverse/UQ approaches is the amount of simulated data being generated and the associated costs of accessing this data.
We shall develop innovative algorithms to overcome this bottleneck by minimizing the amount of simulation data and at the same time hiding the cost of data-movement by overlapping data-access with computation.
#### Projects
* [Deep learning enhanced reduced order models](nsf/sheroze_nsf.md)
* [Managing big simulation data for inverse problems with autoencoders](/cdse/ae_compression)
* [Deep-learning-enhanced POD method](/cdse/pinns_time_dependent_pde)
* [Model constrained DNNs for inverse problems](/cdse/model_constrained)


## Aim 2: Creating strategies to tackle large observation data.
While more observation data generally leads to less uncertain predictions, it
makes the task of inversion and UQ more expensive. We shall develop iterative inversion/UQ methods that utilize only a manageable subset of data during each iteration. This is further enhanced by a streaming computational model that we will create to ensure data scalability while minimizing overall computation.
#### Projects
* [On unifying randomized approaches in inverse problems](/cdse/randomized_approaches)
* [New approach for DNN architecture design](/cdse/layerwise_training)
* [Variational inference with UQ-VAEs](/nsfcareer/year2/uqvae/)
* [Are DNNs actually expressive?](/cdse/active_subspaces_nn_analysis)
* [Active subspaces data-informed approach for inverse problems](/cdse/active_subspaces_inverse_problems)
* [My test project](/test/my_project)


### Publications<a name="publications"></a>

[1] Myers, Aaron and Thiéry, Alexandre H. and Wang, Kainan and Bui-Thanh, Tan. (2021). Sequential ensemble transform for Bayesian inverse problems.  Journal of Computational Physics. 427  (C) 110055. Status = Deposited in NSF-PAR  doi:https://doi.org/10.1016/j.jcp.2020.110055.

[2] Zhang, Wenbo and Rossini, Giovanni and Kamensky, David and Bui‐Thanh, Tan and Sacks, Michael S.. (2021). Isogeometric finite element‐based simulation of the aortic heart valve: Integration of neural network structural material model and structural tensor fiber architecture representations.  International Journal for Numerical Methods in Biomedical Engineering. 37  (4). 

[3] Ambartsumyan, Ilona and Boukaram, Wajih and Bui-Thanh, Tan and Ghattas, Omar and Keyes, David and Stadler, Georg and Turkiyyah, George and Zampini, Stefano. (2020). Hierarchical Matrix Approximations of Hessians Arising in Inverse Problems Governed by PDEs.  SIAM Journal on Scientific Computing. 42  (5) A3397 to A3426. Status = Deposited in NSF-PAR  doi:https://doi.org/10.1137/19M1270367.

[4] Goh, Hwan and Sheriffdeen, Sheroze and Wittmer, Jonathan and Bui-Thanh, Tan (2021). Solving Bayesian Inverse Problems via Variational Autoencoders.  Proceeding of Machine Learning Research, 2nd Annual Conference on Mathematical and Scientific Machine Learning. 145. 

[5] Kang, Shinhoo and Bui-Thanh, Tan. (2021). A scalable exponential-DG approach for nonlinear conservation laws: With application to Burger and Euler equations.  Computer Methods in Applied Mechanics and Engineering. 385  (C) 114031. 

[6] Sheroze Sheriffdeen, Jean C.. (2019). Accelerating PDE-constrained Inverse Solutions with Deep Learning and Reduced Order Models. 
[https://arxiv.org/abs/1912.08864](https://arxiv.org/abs/1912.08864)


[7] Goh, H.. (2020). Solving Forward and Inverse Problems Using Autoencoders. [https://arxiv.org/abs/1912.04212v3](https://arxiv.org/abs/1912.04212v3)

[8] Bui-Thanh, T.. (2021). The Optimality of Bayes' Theorem.  SIAM news. 54 (6).

<!-- ## Accomplishments for Year 1 -->

<!-- We have obtained significant results the for the first year milestones. UT (PI Bui-Thanh and his student) and Utah (PI Sundar and his student) have been meeting weekly (except for weeks that we are busy) and working collaboratively towards meeting the first year goals in parallel. In the following we will discuss the accomplishments from UT and those from Utah can be seen from the annual report from the Utah side. -->

<!-- 1. A partially supported UT student (Sheroze Sheriffdeen) has learned the 3D seismic code to understand how the it works and to generate forward simulation data. -->
<!-- 2. Sheroze has developed a machine learning approach, namely Autoencoder, to compress the forward simulation data for an seismnic inversion in a 3D box. To robustly determine the architecture of the Autoencoder, we use a Bayesian optimization to adaptively and automatically determine the number of layers and the number of neuron on each layers for the Autoencoder neural net (NN). To make the training problem well- posed we imposed an l2 regularization for the weights and biases in the Autoencoder NN. To solve the training optimization, we use the Adam mini-patch stochastic first-order optimizer. We have obtained very promising results with even the standard Autoencoder. We are working on several directions to develop advanced algorithms and results for a paper submitted to SIAM Journal on Scientific Computing.  -->
<!--     1. So far we try to compress the discontinuous and unstructured data through the Autoencoder. It is wellknown that Autoencoder is best for structure data. We are working on mapping discontinuous data to continuous ones and unstructured data to structure ones before compression using a volumetric convolutional Autoencoder. The decompressed data can be mapped back to unstructured and discontinuous data on the fly during the inversion process. -->
<!--     2. At the first step, we have used the squared loss for the training. The better loss function is through the Wasserstein distance and we are the process moving to using this distance. -->
<!--     3. We will investigate on various data size at different inversion parameters to understand the robustness and sensitivty of the approach with respect to the data size and the parameter space. -->
<!--     4. We will study the capability of the method to compress the adjoint simulation data -->
<!--     5. We are then ready to apply our machine learning compression data approach for inversion and study its scalability -->
<!-- 3. Another partially supported UT student (Brad Marvin) has developed a new statistical inversion approach that respects the data compared to the standard Bayesian inversion approach. The method also has the rigorous roof in disintegration theory. We have had significant results for a paper and Brad is writing a paper. -->
<!-- 4. Another partially supported UT student (Jon Wittmer) has extend the new statistical inversion approach to design an interative regularization strategy for inverse problems. He has been also working on developing efficient optimization method for training Neural Networks (NN) and developing robust algorithms for automatically chosing NN architectures. -->

<!-- ## Accomplishments for Year 2 -->

<!-- We have developed a neural-network based data compression technique where the PDE solutions are compressed using a neural network prior to storing, eliminating the need for checkpointing. The solutions are then decompressed using another neural network prior to use in the adjoint problems for computing the gradient and Hessian. We have experimented using data from the 2D heat equation, the 2D wave equation, and the 3D wave equation, with similar results for all. The details are below: -->
<!-- 1. We employ a convolutional auto-encoder where the encoder portion acts to compress the solution at each time-step and the decoder portion decompresses the solution. Using convolutional layers preserves locality of the solution while reducing the number of parameters that need to be learned during the training process. Additionally, the cost of training a neural network to perform the compression and decompression is a one-time up-front cost while the cost of using the trained network in practice is negligible compared to the cost of solving a PDE. -->
<!-- 2. In order to compress to an arbitrary size, we tested a convolutional auto-encoder with a dense layer leading to the compressed state. While the resulting decompressed solutions were similar to the original uncompressed solutions (90\% accuracy on the 2D wave equation), the overall number of parameters to be learned was dominated by the single dense layer (164 million parameters with dense layer vs 161,000 parameters without). Additionally, this architecture has the limitation that it can only be used on data of the exact same size as it was trained. -->
<!-- 3. Another network architecture that was investigated was the fully convolutional auto- encoder. In this architecture, only convolutional layers are used, meaning that we only need to learn the convolutional kernels rather than dense matrix multiplications. This approach has the advantage of being able to be used on arbitrary sized inputs, regardless of the training data size. Additionally, this approach also reduced the number of parameters to be learned during training, further reducing training time, all while improving the accuracy of the decompressed data (95 \% accuracy on the 2D wave equation). The downside is that the solution dimension must be evenly divisible by the compressed dimension. Future work on this architecture is to design a fully-convolutional network where the up/down- sampling process does not need to be evenly divisible by the compression level. -->
<!-- 4. Furthermore, since the fully-convolutional auto-encoder architecture can be used on various input sizes, we developed a procedure whereby the network is first trained on course-grid data (small number of nodes), which is significantly faster than training on fine-grid data (larger number of nodes). This process can be successively carried out over several refinement stages until the final input data size is reached. This "warm-start" training procedure has been experimentally shown to significantly reduce the training time (5 times faster) while not sacrificing the accuracy of the decompressed solution. In some cases, this strategy also leads to an increase in accuracy of the decompressed solution. Further effort will be invested into determining how one should choose the amount of training in each stage in order to maximize the accuracy of the decompressed solution and minimize the training time. -->
<!-- We have also developed a data-driven technique to augment the accuracy of reduced order models by learning their error compared to high-fidelity models and experimental data with the goal of accelerating many-query problems in deterministic inverse problems. This approach reduces the dimensions of the forward and adjoint states. Below are the details and preliminary results that support our approach in accelerating parameter-to-observable maps for an elliptic partial differential equation and a parametrized neutron transport problem. -->
<!-- 1. Learning the error between the full model and the reduced model is a high- dimensional deep learning regression problem whose performance is sensitive to the particular architecture of the deep neural network model. These so-called hyper-parameters of the deep learning model were determined using a Bayesian optimization framework. A Gaussian process surrogate model is employed to parametrize the performance of the neural network as a function of the hyper- parameters. This approach to pick hyper-parameters improves on grid search and random search as the sampling procedure better explores the hyper-parameter space by leveraging information from previous neural network architectures. -->
<!-- 2. Validation of the proposed method is performed using numerical experiments on a steady heat conduction problem and a neutron transport problem. The inverse problem for steady heat conduction is posed as inferring conductivity parameters from sparse observations on a thermal fin. The finite element solution using a coarse mesh was considered as the high fidelity model. An affine decomposition of the resulting stiffness matrix along with projection of the governing equations to a reduced space was used to construct a reduced order model. The error was learned using a deep learning model in a data-driven fashion using simultaneous solves of the high fidelity model and the reduced model for training parameter values sampled from a Gaussian field. The relative error between the ground truth parameters and the parameters computed from solving inverse problems using the high fidelity model, the reduced order model, and the deep learning enhanced reduced order model is computed. The numerical results show that the latter model shows comparable accuracy to the high fidelity model reconstructions while providing computational efficiency similar to that of the reduced order model. -->
<!-- 3. The application of the enhanced reduced order model to neutron transport was verified using comparing the relative prediction error for a quantity of interest (scalar flux over a region of interest) compared to an expensive high-fidelity solution obtained from solving the full transport equations. The reduced order model for this problem is obtained in a physics-informed manner using a diffusion approximation to the collided component of the total transport flux along with energy group collapsing to form discrete energy bins from continuous energy. Further reduction in dimensionality is obtained using a projection of the the governing equations to a reduced space. The numerical experiment for this problem employed the iron-water benchmark, a standard 1-group 2D benchmark for transport solution techniques comprising of three spatial zones. The training parameter set for removal and scattering cross section values was randomly drawn uniformly from predetermined intervals. The neural network was trained to learn the discrepancy between the high-fidelity transport solution and the reduced order models. The average relative prediction error for the validation dataset shows comparable accuracy to the high- fidelity model when the deep learning correction was applied to the physics- informed reduced order models and the projection-based reduced order model. These experiments lend evidence to the ability of the discrepancy function to accurately model reduced order model errors compared to the high-fidelity transport solutions. -->
<!-- Training neural networks to accurately and efficiently solve physical problems purely using data-driven techniques typically require prohibitive amounts of data to be gathered from large number of experimentation scenarios. Mathematical models contain important information regarding the relationships between important quantities of interest. This information can be used to augment the training of neural networks. We formulated a neural network optimization framework that introduces the constraints posed by mathematical models as a penalization term to the neural network loss functions. -->
