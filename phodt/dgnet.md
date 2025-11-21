---
layout: page
title: A Model-Constrained Discontinuous Galerkin Network (DGNet) For Compressible Euler Equations With Out-Of-Distribution Generalization
permalink: /phodt/dgnet
---

### Paper publication

**Publication Link:** [https://www.sciencedirect.com/science/article/pii/S0045782525001847](https://www.sciencedirect.com/science/article/pii/S0045782525001847)

**Citation:**

```bibtex
@article{van2025model,
  title={A model-constrained discontinuous Galerkin Network (DGNet) for compressible Euler equations with out-of-distribution generalization},
  author={Van Nguyen, Hai and Chen, Jau-Uei and Bui-Thanh, Tan},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={440},
  pages={117912},
  year={2025},
  publisher={Elsevier}
}
```

### Methodology

In this work, we developed a machine learning framework to solve shock-type PDEs, in particular, Compressible Euler equations. The core idea is motivated by the dual mesh between Discontinuous Galerkin (DG) method and Graph Neural Network (GNN).

<p align="center">
<img src="/assets/figures/hainguyen/DGNet_dualmesh.png" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b> Figure 1:</b> The dual mesh Discontinuous Galerkin (DG) method and Graph Neural Network (GNN).</figcaption>
</p>

The training data flow is presented as shown in Figure 2. We integrate the data randomization and differentiable solvers to enhance the generalization of neural surrogate models.

<p align="center">
<img src="/assets/figures/hainguyen/DGNet_model.png" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 2:</b> The schematic of DGNet network architecture.</figcaption>
</p>

---

### Numerical results

- We work on 2D Euler equations

$$
\begin{align*}
\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} + \frac{\partial (\rho v)}{\partial y} &= 0 \\
\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} + \frac{\partial (\rho u v)}{\partial y} &= 0 \\
\frac{\partial (\rho v)}{\partial t} + \frac{\partial (\rho u v)}{\partial x} + \frac{\partial (\rho v^2 + p)}{\partial y} &= 0 \\
\frac{\partial E}{\partial t} + \frac{\partial (u(E + p))}{\partial x} + \frac{\partial (v(E + p))}{\partial y} &= 0,
\end{align*}
$$

where $$E$$ is the total energy per unit volume:

$$
E = \frac{p}{\gamma - 1} + \frac{\rho}{2}(u^2 + v^2)
$$

---

#### Problem 1. Airfoil NACA0012

- Training data is generated from Airfoil AoA = 3 and Mach = 0.8, in time interval [0,1.2]s
- Test data is generated from Airfoil AoA = 3 and Mach = 0.8 and Airfoil AoA = 5 and Mach = 1.2 for time interval [0, 7.5]s

<p align="center">
<img src="/assets/figures/hainguyen/2D_Euler_Airfoil_model.png" width="50%" style="display: block; margin: 0 auto; margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 3: (Airfoil)</b> Airfoil configuration AoA-3.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/3D_Euler_Airfoil.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 4: (Airfoil)</b> predictions by DGNet for Airfoil NACA0012 of AoA = 5 and Mach = 1.2.</figcaption>
</p>

---

#### Problem 2. Euler configurations 6 & 12

- Training data is generated from Euler configuration 6 with time interval [0,0.16]s
- Test data is generated from Euler configuration 6 for time interval [0, 0.8]s and Euler configuration 12 for time interval [0, 0.25]s

<p align="center">
<img src="/assets/figures/hainguyen/Euler_config_6_12.png" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 5: (Euler configurations) </b> Information settings.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/3D_Euler_config6_2x2.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 6: (Euler configurations)</b> predictions by DGNet for configuration 6.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/3D_Euler_config12_2x2.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 7: (Euler configurations) </b> predictions by DGNet for configuration 12.</figcaption>
</p>

---

#### Problem 3: Double Mach Reflection

- Training data is generated with time interval [0,0.02]s
- Test data is generated with time interval [0, 0.25]s

<p align="center">
<img src="/assets/figures/hainguyen/2D_Double_Mach_model.png" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 8: (Double Mach Reflection) </b> model.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/2D_Double_Mach.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 9: (Double Mach Reflection) </b> predictions by DGNet.</figcaption>
</p>

---

#### Problem 4. Forward facing corner

- Training data is generated from Model 1 with time interval [0,1]s
- Test data is generated from Model 1 and Model 2 for time interval [0,4]s

<p align="center">
<img src="/assets/figures/hainguyen/2D_Euler_forth_models.png" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 10: (Forward-facing corner) </b> Model 1 and Model 2.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/2D_Euler_forth_same_mesh.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 11: (Forward-facing corner)</b> predictions by DGNet for Model 1.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/2D_Euler_forth_different_mesh.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 12: (Forward-facing corner)</b> predictions by DGNet for Model 2.</figcaption>
</p>

---

#### Problem 5. ScramJet

- Training data is generated from Model 1 with time interval [0,1.6]s
- Test data is generated from Model 1 and Model 2 for time interval [0, 6]s

<p align="center">
<img src="/assets/figures/hainguyen/2D_Euler_scram_jet_models.png" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 13: (ScramJet) </b> Model 1 and Model 2.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/2D_Euler_scram_jet_same_mesh_Mach3.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 14: (ScramJet) </b> predictions by DGNet for Model 1.</figcaption>
</p>

<p align="center">
<img src="/assets/figures/hainguyen/2D_Euler_scram_jet_different_mesh_Mach3.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 15: (ScramJet) </b> predictions by DGNet for Model 2.</figcaption>
</p>

---

#### Problem 6. Hypersonic flow over a sphere cone

- Training data is generated with time interval [0,3e-4]s
- Test data is generated with time interval [0, 1.5e-3]s

<p align="center">
<img src="/assets/figures/hainguyen/2D_Hypersonic.gif" style="margin-bottom: 0px;">
<figcaption align="center" style="margin-top: 2px;"><b>Figure 16: (Hypersonic flow) </b> predictions by DGNet.</figcaption>
</p>
