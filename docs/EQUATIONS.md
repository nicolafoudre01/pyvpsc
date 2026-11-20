# Equations of the strain-rate driven VPSC (as implemented)

Notation: tensors are Mandel vectors (6,) / matrices (6,6), order (xx, yy, zz, xy, xz, yz) with a
factor √2 on the shear components, so that a double contraction `A:B` is a plain dot product and
a fourth-order tensor is a 6×6 matrix.
Grain `g` (weight `w_g`, Σ_g w_g = 1), slip system `s`. `<·> = Σ_g w_g (·)_g`.

## A. Grain law  — `grain_law.py`

| # | equation | code |
|---|---|---|
| (1) | $\tau_s = \mathbf m_s : \boldsymbol\sigma_g$ , $\mathbf m_s = \mathrm{sym}(\mathbf b_s\otimes\mathbf n_s)$ | `resolved_shear` : `S @ sigma` |
| (2) | $\dot\gamma_s = \dot\gamma_{0,s}\,\mathrm{sign}(\tau_s)\,\left\lvert \tau_s/\tau^c_s \right\rvert^{n}$ — **$\dot\gamma_{0,s}$ and $\tau^c_s$ are different for every system** | `slip_rates` |
| (3) | $\mathbf d_g = \sum_s \dot\gamma_s\,\mathbf m_s$  (= `S.T @ gamma_dot`) | `strain_rate` |

Grain stress for a given $\mathbf d$: solve (2)–(3) for $\boldsymbol\sigma$ (damped Newton) — `solve_grain_stress`.

**Per-system parameters.** Every active system $s$ carries its own $\tau^c_s$ and $\dot\gamma_{0,s}$ (`parameters.py`: per family, then per system).
Deactivated systems are simply absent from the sums over $s$ in (3) and (4).
Invariance (used in the tests): $(\tau^c_s,\dot\gamma_{0,s}) \to (\lambda_s\tau^c_s,\ \lambda_s^{n}\dot\gamma_{0,s})$ leaves every $\dot\gamma_s$, hence all stresses, unchanged.
The exponent $n$ is common to all systems (so that Euler's relation $\mathbf M^{tg}\boldsymbol\sigma=n\,\mathbf d$ and $\mathbf d^0=(1-n)\mathbf d$ hold).

## B. Linearization — `linearization.py`

$$\mathbf d \simeq \mathbf M\,\boldsymbol\sigma + \mathbf d^0 \qquad (\mathbf L = \mathbf M^{-1})$$

| # | equation | code |
|---|---|---|
| (4) | $\mathbf M^{sec} = \sum_s w_s\,\mathbf m_s\otimes\mathbf m_s$ , $w_s = \dfrac{\dot\gamma_{0,s}}{\tau^c_s}\left\lvert\dfrac{\tau_s}{\tau^c_s}\right\rvert^{n-1}$  ⇒ $\mathbf d(\boldsymbol\sigma)=\mathbf M^{sec}\boldsymbol\sigma$ | `M_sec = S.T @ (w[:,None]*S)` |
| (5) secant | $\mathbf M = \mathbf M^{sec}$ , $\mathbf d^0 = 0$ | `kind == "secant"` |
| (5) affine | $\mathbf M = \mathbf M^{tg} = n\,\mathbf M^{sec}$ , $\mathbf d^0 = (1-n)\,\mathbf d(\boldsymbol\sigma)$ | `kind == "affine"` |
| (6) | Hutchinson: $\mathbf M \leftarrow \mathbf M + \dfrac{\mathbf J}{3k_{pen}}$ , $\mathbf J=\tfrac13\,\mathbf 1\otimes\mathbf 1$ | `M + J_SPH/(3*kappa)` |

Check: $\mathbf M^{tg}\boldsymbol\sigma + \mathbf d^0 = n\mathbf d + (1-n)\mathbf d = \mathbf d$ (Euler: $\mathbf d$ is homogeneous of degree $n$).
The linearization is exact at the point where it is built.

## C. Self-consistent step — `vpsc.py`

Unknowns: $\boldsymbol\sigma_g$, $\bar{\mathbf L}$, $\boldsymbol\Sigma$. Given: macroscopic strain rate $\mathbf D$.

| # | equation | code (`_iterate`) |
|---|---|---|
| (8) | $\mathbf L_g = \mathbf M_g^{-1}$ ,  $\boldsymbol\sigma_g = \mathbf L_g(\mathbf d_g - \mathbf d^0_g)$ | step 1 and step 5 |
| (10) | $\mathbf Q^* = \mathbf P^{-1} - \bar{\mathbf L}$ ,  $\mathbf P = \mathbf P(\bar{\mathbf L}, \text{shape } a)$ Hill tensor (pyhom `tenseurP`) | step 2 |
| (11) | interaction: $\boldsymbol\sigma_g - \boldsymbol\Sigma = -\mathbf Q^*(\mathbf d_g - \mathbf D)$ | (used to build 12) |
| (12) | from (8)+(11): $(\mathbf L_g+\mathbf Q^*)\,\mathbf d_g = \boldsymbol\Sigma + \mathbf Q^*\mathbf D + \mathbf L_g\mathbf d^0_g$, i.e. $\mathbf d_g = (\mathbf L_g+\mathbf Q^*)^{-1}[\dots]$ | `d = Ainv[g] @ (Sigma + Q@D + L[g]@d0[g])` |
| (13) | impose $\langle\mathbf d_g\rangle=\mathbf D$ : $\boldsymbol\Lambda=\langle(\mathbf L_g+\mathbf Q^*)^{-1}\rangle$ , $\bar{\mathbf L} = \boldsymbol\Lambda^{-1}-\mathbf Q^*$ | `Lam`, `Lbar_new` |
| (14) | $\mathbf X=\langle(\mathbf L_g+\mathbf Q^*)^{-1}\mathbf L_g\mathbf d^0_g\rangle$ , $\boldsymbol\Sigma = \bar{\mathbf L}\mathbf D - \boldsymbol\Lambda^{-1}\mathbf X$ | `X`, `Sigma` |
| (15) | fixed point: $\boldsymbol\sigma_g \leftarrow r\,\boldsymbol\sigma_g^{new} + (1-r)\,\boldsymbol\sigma_g$ ; stop when $\max\lvert\boldsymbol\sigma^{new}-\boldsymbol\sigma\rvert/\max\lvert\boldsymbol\sigma^{new}\rvert<\text{tol}$ | step 6 |
| (0) | start: $\mathbf d_g=\mathbf D$ for all grains (Taylor), $\bar{\mathbf L}=\langle\mathbf L_g\rangle$ | `taylor`, `Lbar = sum(w L)` |

### Derivation of (13)–(14)
Average (12) over grains and use $\langle\mathbf d_g\rangle=\mathbf D$:
$\mathbf D = \boldsymbol\Lambda(\boldsymbol\Sigma+\mathbf Q^*\mathbf D) + \mathbf X$
⇒ $\boldsymbol\Sigma = (\boldsymbol\Lambda^{-1}-\mathbf Q^*)\mathbf D - \boldsymbol\Lambda^{-1}\mathbf X = \bar{\mathbf L}\mathbf D - \boldsymbol\Lambda^{-1}\mathbf X$.
For $\mathbf d^0=0$ (linear) this gives $\boldsymbol\Sigma=\bar{\mathbf L}\mathbf D$ and the classical self-consistent estimate
$\bar{\mathbf L}=\langle(\mathbf L_g+\mathbf Q^*)^{-1}\rangle^{-1}-\mathbf Q^*$ (pyhom `selfconsistent`).

## D. A posteriori checks — `check_equations`

| residual | meaning |
|---|---|
| R1 | $\langle\mathbf d_g\rangle - \mathbf D$  (identity by construction of 13–14) |
| R2 | $\mathbf d_g - [\mathbf d(\boldsymbol\sigma_g) + \mathbf J\boldsymbol\sigma_g/(3k_{pen})]$ : the linearization is exact at the solution (converged fixed point) |
| R3 | $(\boldsymbol\sigma_g-\boldsymbol\Sigma)+\mathbf Q^*(\mathbf d_g-\mathbf D)$ : interaction equation (11) |
| R4 | $\langle\boldsymbol\sigma_g\rangle-\boldsymbol\Sigma$ : mean stress (follows from R1 and R3) |
| R5 | $\bar{\mathbf L} - (\boldsymbol\Lambda^{-1}-\mathbf Q^*)$ : self-consistency of $\bar{\mathbf L}$ |

R1, R3, R4 are algebraic identities (they detect coding errors); R2, R5 measure convergence.

## E. Limits of this implementation
* Fixed texture and fixed spherical (or common ellipsoidal, axes along the sample axes) grains; no hardening.
* $\mathbf D$ is fully imposed (symmetric, traceless). Lateral stresses are not forced to zero: for a textured
  sample they differ slightly from zero (statistical for random textures). Mixed (uniaxial-stress) conditions
  would need an outer loop on the unknown components of $\mathbf D$.
* Tangent linearization: not implemented (definition to be fixed).
