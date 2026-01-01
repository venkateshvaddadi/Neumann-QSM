<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Neumann Network Formulation for QSM</title>

  <!-- MathJax for LaTeX rendering -->
  <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    body {
      font-family: Arial, Helvetica, sans-serif;
      max-width: 900px;
      margin: auto;
      line-height: 1.6;
      padding: 20px;
    }
    h1, h2, h3 {
      color: #2c3e50;
    }
    code {
      background-color: #f4f4f4;
      padding: 2px 4px;
      border-radius: 4px;
    }
    .equation {
      margin: 16px 0;
      text-align: center;
    }
  </style>
</head>

<body>

<h1>Neumann Network Formulation for Quantitative Susceptibility Mapping</h1>

<h2>1. Problem Statement</h2>

<p>
Quantitative Susceptibility Mapping (QSM) can be formulated as a linear inverse problem:
</p>

<div class="equation">
$$
\phi \, \chi = y ,
$$
</div>

<p>
where $\chi$ denotes the magnetic susceptibility distribution, $y$ represents the measured local magnetic field,
and the forward operator $\phi$ is defined as
</p>

<div class="equation">
$$
\phi := \mathcal{F}^{H} D \mathcal{F}.
$$
</div>

<p>
Here, $\mathcal{F}$ and $\mathcal{F}^{H}$ denote the Fourier and inverse Fourier transforms, respectively,
and $D$ is the dipole kernel in the Fourier domain.
</p>

<hr>

<h2>2. Neumann Series Approximation</h2>

<p>
Direct inversion of $\phi$ is ill-posed due to the zeros of the dipole kernel.
Instead, the inverse can be approximated using a truncated Neumann series expansion.
The susceptibility map is expressed as
</p>

<div class="equation">
$$
\chi \approx \sum_{k=0}^{K}
\left( I - \eta \phi^{H}\phi \right)^{k}
\left( \eta \phi^{H} y \right),
$$
</div>

<p>
where $\eta$ is a learnable step size and $K$ denotes the number of unrolled iterations.
</p>

<hr>

<h2>3. Unrolled Network Iterations</h2>

<p>
The Neumann network is implemented through iterative blocks that combine
physics-based data consistency with learned regularization.
</p>

<h3>Initialization</h3>

<p>
The initial estimate is obtained using the adjoint of the forward operator:
</p>

<div class="equation">
$$
\mathbf{B}_0
=
\eta \, \phi^{H} y
=
\eta \, \mathcal{F}^{H}
\left(
D \cdot \mathcal{F}(y)
\right).
$$
</div>

<h3>Data Consistency (Physics Block)</h3>

<p>
To enforce fidelity to the forward model, the normal operator $\phi^{H}\phi$ is applied:
</p>

<div class="equation">
$$
\mathbf{T}_k
=
\mathbf{B}_k
-
\eta \, \mathcal{F}^{H}
\left(
|D|^{2} \cdot \mathcal{F}(\mathbf{B}_k)
\right).
$$
</div>

<h3>Learned Regularization</h3>

<p>
A convolutional neural network (CNN), denoted by $\mathcal{R}_{\theta}$,
acts as a learned regularizer. To improve training stability, normalization is applied:
</p>

<div class="equation">
$$
\mathcal{R}_{\theta}(\mathbf{B}_k)
=
\mathcal{D}_{\theta}
\left(
\frac{\mathbf{B}_k - \mu}{\sigma}
\right)\sigma + \mu ,
$$
</div>

<p>
where $\mu$ and $\sigma$ are the mean and standard deviation computed from the training data.
</p>

<h3>Iterative Update</h3>

<p>
The update rule for each unrolled iteration is given by
</p>

<div class="equation">
$$
\mathbf{B}_{k+1}
=
\mathbf{T}_k
-
\eta \, \mathcal{R}_{\theta}(\mathbf{B}_k).
$$
</div>

<hr>

<h2>4. Final Reconstruction</h2>

<p>
Following the Neumann series formulation, the final susceptibility map is obtained by
aggregating all intermediate estimates:
</p>

<div class="equation">
$$
\hat{\chi}
=
\sum_{k=0}^{K}
\mathbf{B}_k .
$$
</div>

</body>
</html>
