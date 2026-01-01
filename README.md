Below is a **clean, GitHub-ready `README.md` version** of your content.
It uses **GitHub-supported Markdown + LaTeX math** (`$$ ... $$` and `$...$`), which renders correctly on **GitHub, GitLab, and most Markdown viewers with MathJax enabled**.

You can **copy–paste this directly** into `README.md`.

---

# Neumann Network Formulation for Quantitative Susceptibility Mapping

## Problem Statement

Quantitative Susceptibility Mapping (QSM) can be formulated as a linear inverse problem:

$$
\phi , \chi = y ,
$$

where

* $\chi$ denotes the magnetic susceptibility distribution,
* $y$ represents the measured local magnetic field, and
* the forward operator $\phi$ is defined as

$$
\phi := \mathcal{F}^{H} D \mathcal{F}.
$$

Here, $\mathcal{F}$ and $\mathcal{F}^{H}$ denote the Fourier and inverse Fourier transforms, respectively, and $D$ is the dipole kernel in the Fourier domain.

---

## Neumann Series Approximation

Direct inversion of $\phi$ is ill-posed due to the zeros of the dipole kernel. Instead, the inverse can be approximated using a truncated Neumann series expansion. The susceptibility map is expressed as

$$
\chi \approx \sum_{k=0}^{K}
\left( I - \eta \phi^{H}\phi \right)^{k}
\left( \eta \phi^{H} y \right),
$$

where

* $\eta$ is a learnable step size, and
* $K$ denotes the number of unrolled iterations.

---

## Unrolled Network Iterations

The Neumann network is implemented through iterative blocks that combine physics-based data consistency with learned regularization.
Sure 👍
Below is a **clean, GitHub-ready Markdown version** with **proper math rendering** (MathJax/KaTeX compatible) and **corrected equation formatting**.


### Initialization

The initial estimate is obtained using the adjoint of the forward operator:

```math
\mathbf{B}_0
=
\eta \, \phi^{H} y
=
\eta \, \mathcal{F}^{H}
\left(
D \cdot \mathcal{F}(y)
\right)
```

---



## Data Consistency (Physics Block)

To enforce fidelity to the QSM forward model, the **normal operator**
(\phi^{H}\phi) is applied at each iteration.

The data-consistency update is defined as:

```math
\mathbf{T}_k
=
\mathbf{B}_k
-
\eta \, \mathcal{F}^{H}
\left(
|D|^{2} \cdot \mathcal{F}(\mathbf{B}_k)
\right)
```

### Explanation

* (\mathbf{B}_k): Current estimate of the susceptibility-related field
* (\eta): Learnable step size
* (\mathcal{F}(\cdot)), (\mathcal{F}^{H}(\cdot)): Fourier and inverse Fourier transforms
* (D): Dipole kernel in the Fourier domain
* (|D|^2): Magnitude-squared dipole kernel, corresponding to the normal operator
* (\mathbf{T}_k): Intermediate estimate after enforcing data consistency

This step ensures that the current estimate remains consistent with the physical QSM forward model before applying learned regularization.

---

