# Strategy for estimating reduced Ruelle–Pollicott resonances in PlaSim

## Objective

The goal is to estimate slow relaxation and oscillatory structures from the
PlaSim–LSG integrations using field-valued temperature observations. The
initial observation is the annual Northern Hemisphere zonal-mean surface
temperature,

$$
Y_t = h(X_t)
    = \left(T_s(\varphi_1,t),\ldots,T_s(\varphi_{16},t)\right),
$$


where the latitudes are the Northern Hemisphere points of the T21 Gaussian
grid. The method should identify resonances that describe the decay of
correlations in this observed state and determine how those resonances change
with the solar parameter `mu`.

The immediate target is the spectrum of a reduced conditional-expectation
operator, not the full deterministic Ruelle–Pollicott spectrum of PlaSim.

## Full Koopman spectrum versus reduced resonances

For deterministic dynamics preserving an invariant measure \(\mu\), the
Koopman operator

$$
U^\tau f = f\circ\Phi^\tau
$$

is unitary on \(L^2_\mu\) when the dynamics are invertible. Its spectrum lies
on the unit circle; decay of correlations is generally associated with
continuous spectrum rather than isolated eigenvalues inside the disk
[[1]](https://arxiv.org/abs/1507.02228).

Full deterministic Ruelle–Pollicott (RP) resonances arise when transfer or
Koopman operators are studied on suitable anisotropic spaces, or under a
stochastic regularization whose zero-noise limit is known to converge. An
interior eigenvalue produced by finite-rank EDMD, kernel smoothing, or finite
sampling is therefore not automatically a full-system RP resonance
[[1]](https://arxiv.org/abs/1507.02228).

For the partial observation \(Y_t=h(X_t)\), define instead

$$
T_\tau f(y)
=
\mathbb{E}\!\left[f(Y_{t+\tau})\mid Y_t=y\right].
$$

Equivalently, \(T_\tau\) is the full Koopman evolution followed by conditional
projection onto functions of the observed field. It is generally nonunitary
because ocean, sea-ice, atmospheric, and other variables are unobserved. Its
interior eigenvalues have a principled interpretation as reduced RP
resonances, subject to tests for memory, sampling error, and numerical
convergence [[3]](https://arxiv.org/abs/1912.03170).

Thus the intended result is:

> Reduced Ruelle–Pollicott resonances of PlaSim–LSG conditioned on the
> Northern Hemisphere zonal surface-temperature observation.

## Role of diffusion maps

Diffusion maps will construct a smooth, invariant-measure-adapted Galerkin
basis. Their eigenvalues will not be interpreted as the dynamical resonances.
The dynamics enter separately through time-shifted sample pairs.

Following Berry, Giannakis, and Harlim
[[2]](https://arxiv.org/abs/1411.5069), diffusion maps approximate an auxiliary
elliptic generator

$$
\widehat L = \Delta - \nabla U\cdot\nabla,
\qquad U=-\log p_{\mathrm{eq}},
$$

whose invariant density is the observed sampling density
\(p_{\mathrm{eq}}\). Its eigenfunctions \(\{\varphi_j\}\):

- form an approximately orthonormal basis in
  \(L^2(p_{\mathrm{eq}})\);
- are ordered by smoothness;
- adapt to the intrinsic data manifold rather than the 16-dimensional ambient
  coordinate system;
- provide a lower-variance basis for estimating the dynamical shift operator
  [[2]](https://arxiv.org/abs/1411.5069).

This separates geometry from dynamics:

$$
\text{diffusion maps} \longrightarrow \text{smooth basis},
$$

$$
\text{time-shifted observations} \longrightarrow
\text{reduced dynamical operator}.
$$

## Field-space metric and preprocessing

For two zonal-temperature anomalies \(x\) and \(y\), use the Gaussian-area
metric

$$
\lVert x-y\rVert_w^2
=
\sum_{i=1}^{16} w_i(x_i-y_i)^2,
$$

where \(w_i\) are the Northern Hemisphere Gaussian quadrature weights,
renormalized to sum to one.

The initial analysis should:

1. use `zonal_surface_temperature`;
2. retain only `lat > 0`;
3. retain years after the simulation-specific transient threshold;
4. subtract the stationary temporal mean at each latitude;
5. preserve physical temperature amplitudes rather than standardizing each
   latitude independently.

Per-latitude variance normalization may be examined as a sensitivity test, but
it changes the physical metric by giving low-variance and high-variance
latitudes equal influence.

## Variable-bandwidth diffusion-map normalization

The observed invariant density is strongly nonuniform. A fixed Gaussian kernel
can overresolve the densely sampled centre of the attractor and poorly resolve
rare excursions. Use a variable-bandwidth kernel of the form

$$
K_\epsilon(x,y)
=
\exp\!\left[
-\frac{\lVert x-y\rVert_w^2}
{4\epsilon\,[q_\epsilon(x)q_\epsilon(y)]^\beta}
\right].
$$

In the normalization convention used by Berry, Giannakis, and Harlim, choose

$$
\beta=-\frac12,
\qquad
\alpha=-\frac d4,
$$

where \(d\) is the estimated intrinsic dimension. This choice yields the
auxiliary elliptic operator with invariant measure equal to the sampling
measure [[2]](https://arxiv.org/abs/1411.5069). The values of \(\alpha\) and
\(\beta\) are convention-dependent and must not be mixed with fixed-bandwidth
Coifman–Lafon normalization parameters.

The implementation should follow the sequence in
[[2]](https://arxiv.org/abs/1411.5069):

1. estimate a local scale from nearest-neighbour distances;
2. estimate the sampling density \(q_\epsilon\);
3. estimate the intrinsic dimension from the scaling of the kernel sum;
4. select the global bandwidth from the maximum logarithmic scaling slope;
5. construct the two density normalizations;
6. solve the symmetric conjugate eigenproblem;
7. retain the first \(m\) smooth basis functions.

The basis dimension \(m\) and global bandwidth remain convergence parameters.

## Galerkin estimate of the reduced dynamical operator

For a lag of \(\ell\) annual samples, estimate

$$
G_{ij}
=
\frac{1}{N-\ell}
\sum_{t=1}^{N-\ell}
\varphi_i(Y_t)\varphi_j(Y_t),
$$

and

$$
A^{(\ell)}_{ij}
=
\frac{1}{N-\ell}
\sum_{t=1}^{N-\ell}
\varphi_i(Y_t)\varphi_j(Y_{t+\ell}).
$$

The finite-dimensional reduced operator is

$$
K_\ell = G^\dagger A^{(\ell)}.
$$

For a well-resolved diffusion basis, \(G\) should be close to the identity,
but it should still be calculated and regularized explicitly. Singular values
of \(G\) and the retained rank must be reported.

Let \(\zeta_k(\ell)\) be an eigenvalue of \(K_\ell\). Its corresponding rate is

$$
\lambda_k(\ell)
=
\frac{\log|\zeta_k(\ell)|}{\ell\,\Delta t}
+ i\frac{\arg\zeta_k(\ell)}{\ell\,\Delta t},
$$

with \(\Delta t=1\) model year for the current datasets. Frequencies are
subject to the Nyquist ambiguity
\(|\operatorname{Im}\lambda|<\pi/(\ell\Delta t)\).

## Essential lag-consistency test

Because an observed field is generally non-Markovian, the family of reduced
operators need not satisfy a semigroup law. Estimate each lagged operator
directly for

$$
\ell\in\{1,2,3,5,10\}\ \text{years}.
$$

Do not obtain longer-lag operators merely by taking powers of the one-year
matrix. A credible reduced resonance should produce approximately the same
continuous-time rate \(\lambda_k(\ell)\) across a useful interval of lags.

Eigenvalues should be matched between analyses using one-to-one matching in
the complex plane. When eigenvalues cluster, cross, or form complex-conjugate
pairs, compare the associated invariant subspaces using principal angles rather
than assigning physical meaning to a fixed eigenvalue index.

Lag inconsistency is evidence of unresolved memory in the chosen observation.
It is not merely a numerical error
[[1]](https://arxiv.org/abs/1507.02228)
[[3]](https://arxiv.org/abs/1912.03170).

## Correlation and spectral validation

An estimated resonance is useful only if it explains measured variability.
Project physically relevant observables onto the left and right eigenfunctions
and reconstruct:

- the autocorrelation of Northern Hemisphere mean surface temperature;
- the autocorrelation of the Northern meridional temperature contrast;
- their cross-correlation;
- their power spectral densities.

Compare each reconstruction with a direct estimate from the stationary time
series. This is the main closure test advocated by the theory of reduced RP
resonances [[3]](https://arxiv.org/abs/1912.03170).

For the reduced phase space, use the Northern-only quantities

$$
\langle T_s\rangle_N
=
\langle T_s\rangle_{0^\circ\text{--}90^\circ N},
$$

and

$$
\Delta T_N^{\mathrm{NH}}
=
\langle T_s\rangle_{0^\circ\text{--}30^\circ N}
-
\langle T_s\rangle_{30^\circ\text{--}90^\circ N}.
$$

This differs from the existing scalar diagnostic
`northern_polar_temperature_gradient`, whose tropical term covers
30 degrees S through 30 degrees N. The Northern-only definition is preferable
when the operator state contains only Northern Hemisphere temperatures.

## Numerical sensitivity tests

For each `mu`, test at least:

### Diffusion-map construction

- fixed versus variable bandwidth;
- bandwidth multipliers around the automatically selected value;
- intrinsic-dimension estimate;
- nearest-neighbour count used for the local scale;
- number of diffusion basis functions, initially
  \(m\in\{25,50,100,200\}\);
- treatment of isolated samples and poorly supported regions.

### Dynamical operator

- lag \(\ell\);
- regularization and retained rank of \(G\);
- full, half, and quarter trajectory lengths;
- blocked bootstrap samples, with block length longer than the dominant
  correlation time.

### Spectral results

- eigenvalue movement in the complex plane;
- stability of relaxation times and oscillation periods;
- invariant-subspace stability;
- correlation and PSD reconstruction error;
- eigenfunction stability under Nyström evaluation on withheld data.

The first five current KDMD eigenvalues are not equally robust: the leading
real eigenvalue is comparatively stable, while subsequent modes reorder and
switch between real and complex-pair representations. The diffusion-map
analysis should therefore retain the same cautious mode-matching and subspace
tests.

This type of numerical robustness analysis follows the PlaSim transfer-operator
study in [[1]](https://arxiv.org/abs/1507.02228).

## Delay coordinates

Delay embedding should not be the default first step. It creates a tradeoff:

$$
\text{few delays}
\longrightarrow
\text{strong projection and memory},
$$

$$
\text{many delays}
\longrightarrow
\text{a more complete, increasingly deterministic state representation}.
$$

In the limit of a sufficiently informative deterministic embedding, the
problem approaches the unitary \(L^2_\mu\) Koopman problem rather than a
strongly contracting reduced operator.

Begin with no delays and diagnose memory through lag inconsistency. If needed,
repeat the analysis for a small number of delays, for example
\(Q\in\{2,5,10\}\), while recognizing that changing \(Q\) changes the reduced
state space and therefore the reduced resonances being estimated.

Delay-coordinate kernel constructions for high-dimensional fields are also
developed in [[5]](https://arxiv.org/abs/2606.06728).

## Physical interpretation of eigenfunctions

For every robust leading eigenfunction:

1. evaluate it along the complete stationary trajectory;
2. identify samples in its upper and lower quantiles;
3. composite the original zonal-temperature fields over those samples;
4. subtract the two composites to obtain an interpretable spatial pattern;
5. compare that pattern with
   \(\langle T_s\rangle_N\), \(\Delta T_N^{\mathrm{NH}}\), sea ice, and AMOC;
6. test whether its variance and correlation time change systematically with
   `mu`.

The use of diffusion coordinates as nonlinear slow observables, together with
Nyström evaluation and physical field composites, is closely related to
[[4]](https://doi.org/10.1103/l2v2-xndy).

For a complex pair, analyze magnitude and phase or phase composites. The real
and imaginary components can be rotated by an arbitrary complex phase and
should not be assigned separate physical meanings without phase alignment.

## Implementation sequence

### Phase 1: stationary `mu=1367` prototype

1. Implement Gaussian-area-weighted distances.
2. Implement the variable-bandwidth diffusion-map normalization.
3. estimate the intrinsic dimension and bandwidth;
4. construct diffusion bases for several values of \(m\);
5. estimate \(K_\ell\) independently at several lags;
6. plot and match the leading rates;
7. validate Northern temperature and gradient correlations and PSDs.

### Phase 2: physical interpretation

1. Plot eigenfunctions in
   \((\langle T_s\rangle_N,\Delta T_N^{\mathrm{NH}})\) space.
2. Produce positive-minus-negative temperature composites.
3. Quantify projections onto the two reduced temperature coordinates.
4. Relate robust modes to sea-ice and AMOC diagnostics.

### Phase 3: continuation in `mu`

Repeat the converged analysis for each stationary regime. Track resonances and
invariant subspaces between neighbouring parameter values. Do not force a
single mode correspondence across simulations with qualitatively different
dynamics, such as equilibria, limit cycles, and recurrent transitions.

### Phase 4: extended observed states

If temperature-only operators fail the lag and reconstruction tests, compare:

- temperature with a small delay embedding;
- surface and 2-metre temperature together;
- temperature plus sea-ice diagnostics;
- temperature plus AMOC;
- joint temperature, sea-ice, and AMOC observations.

Each extension defines a different reduced operator and must be documented as
such.

## Claims to avoid

Until a stronger convergence argument is available, do not claim that:

- diffusion-map eigenvalues are PlaSim RP resonances;
- arbitrary interior EDMD eigenvalues are physical decay rates;
- a single-lag spectrum demonstrates Markovian reduced dynamics;
- a smooth phase-space interpolation provides new dynamical information;
- reduced resonances are the full deterministic PlaSim resonance spectrum;
- eigenfunctions with unstable indices across hyperparameters represent fixed
  physical modes.

## References

1. Tantet, A., Lucarini, V., Lunkeit, F., and Dijkstra, H. A. (2015).
   [*Crisis of the Chaotic Attractor of a Climate Model: A Transfer Operator
   Approach*](https://arxiv.org/abs/1507.02228), arXiv:1507.02228. This is the
   direct PlaSim precedent based on scalar and two-dimensional time-series
   reductions, Ulam transition matrices, and robustness tests in lag, grid
   resolution, and trajectory length.

2. Berry, T., Giannakis, D., and Harlim, J. (2015).
   [*Nonparametric forecasting of low-dimensional dynamical
   systems*](https://arxiv.org/abs/1411.5069), arXiv:1411.5069. This provides
   the variable-bandwidth diffusion-map basis and Galerkin approximation of
   the time-shift semigroup used in the proposed method.

3. Chekroun, M. D., Tantet, A., Dijkstra, H. A., and Neelin, J. D. (2020).
   [*Ruelle–Pollicott Resonances of Stochastic Systems in Reduced State Space.
   Part I: Theory*](https://arxiv.org/abs/1912.03170), arXiv:1912.03170. This
   gives the conditional-expectation interpretation of reduced resonances and
   their validation through correlation functions and power spectra.

4. Lohmann, J., and Gottwald, G. A. (2025).
   [*Choosing observables that capture critical slowing down before tipping
   points: A Fokker–Planck operator
   approach*](https://doi.org/10.1103/l2v2-xndy), *Physical Review E*, 112,
   064204. This motivates diffusion coordinates as nonlinear observables of
   slow relaxation, Nyström evaluation, and physical field composites.

5. Froyland, G., Giannakis, D., and Peters, N. (2026).
   [*Data-driven methods for computation of optimal linear response in
   high-dimensional dynamical systems*](https://arxiv.org/abs/2606.06728),
   arXiv:2606.06728. This develops high-dimensional field kernels, delay
   embeddings, normalized kernel Markov operators, and spatial interpretation;
   kernel-regularized eigenvalues are not automatically full deterministic RP
   resonances.
