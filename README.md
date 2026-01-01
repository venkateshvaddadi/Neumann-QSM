
## Neumann Network Formulation for Quantitative Susceptibility Mapping

### Problem Statement

Quantitative Susceptibility Mapping (QSM) can be formulated as a linear inverse problem:

$$
\phi \chi = y
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

$$ \chi \approx \sum_{k=0}^{K} \left( I - \eta \phi^{H}\phi \right)^{k} \left( \eta \phi^{H} y \right) $$

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
\mathbf{B}_0 = \eta  \phi^{H} y
$$

Using the definition of the QSM forward operator  
$\phi = \mathcal{F}^{H} D \mathcal{F}$,  
the initialization can be written explicitly as:

$$
\mathbf{B}_0 = \eta  \mathcal{F}^{H} \left( D \cdot \mathcal{F}(y) \right)
$$

---




## Data Consistency (Physics Block)

To enforce fidelity to the QSM forward model, the **normal operator**
$\phi^{H}\phi$ is applied at each unrolled iteration.

Given the definition of the QSM forward operator

$$ \phi = \mathcal{F}^{H} D \mathcal{F} $$

the corresponding normal operator becomes

$$\phi^{H}\phi(\mathbf{B}_k)= \mathcal{F}^{H} \left( |D|^{2} \cdot \mathcal{F}(\mathbf{B}_k) \right)$$

The **data consistency update** at iteration (k) is then expressed as


$$\mathbf{T}_k=\mathbf{B}_k-\eta \mathcal{F}^{H} \left( |D|^{2} \cdot \mathcal{F}(\mathbf{B}_k) \right)$$
where:

* $\mathbf{B}_k$ denotes the current susceptibility estimate,
* $D$ is the dipole kernel in the Fourier domain,
* $\mathcal{F}$ and $\mathcal{F}^{H}$ represent the Fourier and inverse Fourier transforms,
* $\eta$ is a learnable step size controlling the data consistency strength.

---


## Learned Regularization

A convolutional neural network (CNN), denoted by $\mathcal{R}_{\theta}$, is used as a **learned regularizer** within each unrolled iteration.
To improve training stability and convergence, the input is normalized using the mean and standard deviation computed from the training data.

The learned regularization is defined as:

$$ \mathcal{R}_{\theta}(\mathbf{B}_k) = \mathcal{D}_{\theta} \left( \frac{\mathbf{B}_k - \mu}{\sigma} \right)\sigma + \mu $$

where:

* $\mathcal{D}_{\theta}$ denotes the CNN with learnable parameters $\theta$,
* $\mu$ and $\sigma$ are the mean and standard deviation of the training data,
* $\mathbf{B}_k$ is the current susceptibility estimate at iteration $k$.

---

## Iterative Update

The estimate is updated by combining the **physics-based data consistency term** with the **learned regularization term**:

$$
\mathbf{B}_{k+1}= \mathbf{T}_k- \eta , \mathcal{R}_{\theta}(\mathbf{B}_k)
$$

where:

* $\mathbf{T}_k$ is the intermediate result after applying the data consistency operation,
* $\eta$ is a learnable step size,
* $\mathcal{R}_{\theta}(\mathbf{B}_k)$ represents the CNN-based regularization.

---



