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




### Inline math
```markdown
The operator is defined as $\phi = \mathcal{F}^H D \mathcal{F}$.
````

---

## 🧠 Final Output (as it appears)

[
\mathbf{B}_0
============

# \eta , \phi^{H} y

\eta , \mathcal{F}^{H}
\left(
D \cdot \mathcal{F}(y)
\right)
]

---

If you want, I can also:

* ✅ Convert the **entire Neumann-QSM section** into one polished `README.md`
* 📊 Add an **ASCII / Mermaid flow diagram** for GitHub
* 🧩 Provide a **pseudo-code block** that renders nicely in Markdown
* 🌐 Prepare a **GitHub Pages–ready version**

Just tell me 👍


### Data Consistency (Physics Block)

To enforce fidelity to the forward model, the normal operator $\phi^{H}\phi$ is applied:

$$
\mathbf{T}_k
============

## \mathbf{B}_k

\eta , \mathcal{F}^{H}
\left(
|D|^{2} \cdot \mathcal{F}(\mathbf{B}_k)
\right).
$$

---

### Learned Regularization

A convolutional neural network (CNN), denoted by $\mathcal{R}_{\theta}$, acts as a learned regularizer. To improve training stability, normalization is applied:

$$
\mathcal{R}_{\theta}(\mathbf{B}_k)
==================================

\mathcal{D}_{\theta}
\left(
\frac{\mathbf{B}_k - \mu}{\sigma}
\right)\sigma + \mu ,
$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed from the training data.

---

### Iterative Update

The update rule for each unrolled iteration is given by

$$
\mathbf{B}_{k+1}
================

## \mathbf{T}_k

\eta , \mathcal{R}_{\theta}(\mathbf{B}_k).
$$

---

## Final Reconstruction

Following the Neumann series formulation, the final susceptibility map is obtained by aggregating all intermediate estimates:

$$
\hat{\chi}
==========

\sum_{k=0}^{K}
\mathbf{B}_k .
$$

---

## Notes on Rendering Math on GitHub

* GitHub supports LaTeX math using `$$ ... $$` (block math) and `$...$` (inline math).
* Math rendering is powered by **MathJax** and works automatically in `README.md`.
* Equation labels (`\label{}`) are omitted, as GitHub does not support equation referencing.

---

### ✅ This format is:

* ✔ GitHub-compatible
* ✔ Clean and readable
* ✔ Paper–code consistent
* ✔ Ideal for QSM / Neumann / MoDL repositories

If you want, I can also:

* Add an **algorithm block** in Markdown
* Add a **diagram + caption**
* Write a **“Theory” vs “Implementation” section**
* Convert this into **GitHub Pages / Jekyll** math format

Just tell me 👍
