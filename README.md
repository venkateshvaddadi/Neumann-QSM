Absolutely 👍
Below is a **complete, clean Markdown (`README.md`) version** of your **entire LaTeX document**, written so that:

* ✅ It renders correctly on **GitHub**
* ✅ All equations use **GitHub-supported LaTeX math** (`$$ ... $$`)
* ✅ Headings, structure, and wording are preserved
* ✅ No LaTeX-only commands (`\section`, `\paragraph`, `\label`) remain

You can **copy–paste this directly into `README.md`**.

---

# Neumann-QSM

**Author:** Venkatesh Vaddadi
**Date:** January 2026

---

## Neumann Network Formulation for Quantitative Susceptibility Mapping

### Problem Statement

Quantitative Susceptibility Mapping (QSM) can be formulated as a linear inverse problem:

$$
\phi , \chi = y
$$

where

* $\chi$ denotes the magnetic susceptibility distribution,
* $y$ represents the measured local magnetic field, and
* the forward operator $\phi$ is defined as:

$$
\phi := \mathcal{F}^{H} D \mathcal{F}
$$

Here, $\mathcal{F}$ and $\mathcal{F}^{H}$ denote the Fourier and inverse Fourier transforms, respectively, and $D$ is the dipole kernel in the Fourier domain.

---

### Neumann Series Approximation

Direct inversion of $\phi$ is ill-posed due to the zeros of the dipole kernel. Instead, the inverse can be approximated using a truncated Neumann series expansion. The susceptibility map is expressed as:

$$
\chi \approx \sum_{k=0}^{K}
\left( I - \eta \phi^{H}\phi \right)^{k}
\left( \eta \phi^{H} y \right)
$$

where

* $\eta$ is a learnable step size, and
* $K$ denotes the number of unrolled iterations.

---

### Unrolled Network Iterations

The Neumann network is implemented through iterative blocks that combine
physics-based data consistency with learned regularization.

---

### Initialization

The initial estimate is obtained using the adjoint of the forward operator:

$$
\mathbf{B}_0 = \eta \, \phi^{H} y
$$

Using the definition of the QSM forward operator  
$\phi = \mathcal{F}^{H} D \mathcal{F}$,  
the initialization can be written explicitly as:

$$
\mathbf{B}_0
=
\eta \, \mathcal{F}^{H}
\left(
D \cdot \mathcal{F}(y)
\right)
$$

---


#### Data Consistency (Physics Block)

To enforce fidelity to the forward model, the normal operator $\phi^{H}\phi$ is applied:

$$
\mathbf{T}_k
============

## \mathbf{B}_k

\eta , \mathcal{F}^{H}
\left(
|D|^{2} \cdot \mathcal{F}(\mathbf{B}_k)
\right)
$$

---

#### Learned Regularization

A convolutional neural network (CNN), denoted by $\mathcal{R}_{\theta}$, acts as a learned regularizer. To improve training stability, normalization is applied:

$$
\mathcal{R}_{\theta}(\mathbf{B}_k)
==================================

\mathcal{D}_{\theta}
\left(
\frac{\mathbf{B}_k - \mu}{\sigma}
\right)\sigma + \mu
$$

where

* $\mu$ and $\sigma$ are the mean and standard deviation computed from the training data.

---

#### Iterative Update

The update rule for each unrolled iteration is given by:

$$
\mathbf{B}_{k+1}
================

## \mathbf{T}_k

\eta , \mathcal{R}_{\theta}(\mathbf{B}_k)
$$

---

### Final Reconstruction

Following the Neumann series formulation, the final susceptibility map is obtained by aggregating all intermediate estimates:

$$
\hat{\chi}
==========

\sum_{k=0}^{K}
\mathbf{B}_k
$$

