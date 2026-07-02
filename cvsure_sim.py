"""
Core simulation library for "From Cross-Validation to SURE".

Validates the paper's key approximations numerically. The diagnostics use the
manuscript notation throughout (Delta_IF, Delta_CV, Delta_tune, R_n, R, Delta_R);
the raw per-replication accumulators kept in the cell *.npz files, from which the
diagnostics are formed, are listed in parentheses:

  * Lemma 2  (influence-function approximation)   -> Delta_IF   (Delta_IF_num/Delta_IF_den)
  * Lemma 4  (uniform convergence of CV to SURE)   -> Delta_CV   (Delta_CV_num/Delta_CV_den)
  * Lemma 5  (convergence of the tuning parameter) -> Delta_tune (excess, sure_tuned_loss)
  * Theorem 1 / Corollary 2 (convergence of risk)  -> R_n, R, Delta_R (Lbar_min)

Design choices that keep the code faithful AND simple:
  * All designs are built directly in the normalized coordinates of
    Assumption 2 (limiting Hessian H = I) and with score variance Sigma = I.
    Anisotropy (the source of multi-modal SURE) is carried by the penalty
    matrix A, exactly as in the multimodality figure of the paper.
  * Sigma and theta0 are KNOWN (we control the DGP), so SURE uses the true
    Sigma and the influence function uses the true theta0. This isolates the
    approximation each lemma claims, rather than confounding it with the
    estimation of Sigma.

Only numpy / pandas / matplotlib are used.
"""
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------
# Numerically stable scalar helpers
# --------------------------------------------------------------------------
def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))

def softplus(x):                                # log(1 + e^x), overflow-safe
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


# ==========================================================================
# 1. PENALTIES
#    Each penalty is diagonal in coordinates, with diagonal A = diag(a).
#    It exposes the limiting shrinkage map g^lambda and its Jacobian,
#    the proximal operator (for the finite-sample solver), and SURE.
# ==========================================================================
class Ridge:
    name = "Ridge"

    def __init__(self, a):
        self.a = np.asarray(a, float)          # diagonal of A (A is pos. def.)

    def g(self, theta, lam):                   # g^lambda(theta) = thetahat^lambda - theta
        if np.isinf(lam):
            return -theta                      # full shrinkage at lambda = infinity
        return -theta * lam / (self.a + lam)

    def dg(self, theta, lam):                  # diagonal of d g^lambda / d theta
        if np.isinf(lam):
            return -np.ones_like(theta)
        return -lam / (self.a + lam)

    def prox(self, v, step, lam):              # prox of (step * lam * pi)
        if np.isinf(lam):
            return np.zeros_like(v)
        return v / (1.0 + step * lam / self.a)

    def lam_grid(self, n_grid):
        # Compactified Lambda = R+ u {inf}; uniform in s = lam/(1+lam) in [0,1].
        s = np.linspace(0.0, 1.0, n_grid)
        lam = np.full(n_grid, np.inf)
        interior = s < 1.0
        lam[interior] = s[interior] / (1.0 - s[interior])
        return lam


class Lasso:
    name = "Lasso"

    def __init__(self, a):
        self.a = np.asarray(a, float)          # A = diag(a); pi(theta) = ||A^{-1} theta||_1

    def g(self, theta, lam):
        tau = lam / self.a
        soft = np.sign(theta) * np.maximum(np.abs(theta) - tau, 0.0)
        return soft - theta

    def dg(self, theta, lam):
        tau = lam / self.a
        # d g = d(soft-threshold) - I ; convention d g = 0 at kinks (|theta| = tau)
        return np.where(np.abs(theta) > tau, 0.0, -1.0)

    def prox(self, v, step, lam):
        tau = step * lam / self.a
        return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

    def lam_grid(self, n_grid, lam_max=3.0):
        return np.linspace(0.0, lam_max, n_grid)


def sure(penalty, thetahat, lam, sigma_diag):
    """SURE(lambda, thetahat, Sigma) for diagonal Sigma = diag(sigma_diag).

    Follows the manuscript definition, which carries an overall factor 1/2:
        SURE = 1/2 [ trace(Sigma) + ||g^lambda||^2 + 2 trace(grad g^lambda . Sigma) ].
    With this factor SURE is an unbiased estimate of the risk
    E[1/2 ||theta^lambda - theta0||^2] (Lemma 3 / Theorem 1), i.e. on the same
    (loss) scale as n-fold CV, so CV is compared to SURE directly.
    """
    g = penalty.g(thetahat, lam)
    dg = penalty.dg(thetahat, lam)
    return 0.5 * (sigma_diag.sum() + g @ g + 2.0 * (dg * sigma_diag).sum())


# ==========================================================================
# 2. FISTA  (accelerated proximal gradient) for general penalized ERM
#    Minimises  L_n(theta) + lambda * pi(theta)  given a smooth gradient and
#    a proximal operator. Warm starts make the leave-one-out refits cheap.
# ==========================================================================
def fista(grad, prox, L, x0, max_iter=400, tol=1e-7):
    x = x0.copy()
    z = x0.copy()
    t = 1.0
    step = 1.0 / L
    for _ in range(max_iter):
        x_prev = x
        x = prox(z - step * grad(z), step)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x + ((t - 1.0) / t_new) * (x - x_prev)
        t = t_new
        if np.linalg.norm(x - x_prev) <= tol * (np.linalg.norm(x) + 1e-12):
            break
    return x


# ==========================================================================
# 3. MODELS
#    Each model knows how to simulate data, fit the (unpenalized) ERM, form
#    the influence-function proxy theta-tilde, evaluate the smooth gradient
#    (for FISTA), the pointwise loss (for CV) and the out-of-sample loss Lbar.
# ==========================================================================
class LinearModel:
    """Linear regression with quadratic loss l(beta,z) = (y - w.beta)^2.

    DGP chosen so that H = 2 E[ww'] = I and Sigma = I:
      W ~ N(0, 0.5 I),  U ~ N(0, 0.5),  Y = W.theta0/sqrt(n) + U.
    Then the out-of-sample loss is exactly Lbar_n(theta,theta0) = 0.5||theta-theta0||^2.
    """
    name = "linear"
    has_closed_form_ridge = True

    def generate(self, n, theta0, rng):
        p = theta0.size
        W = rng.normal(0.0, np.sqrt(0.5), size=(n, p))
        U = rng.normal(0.0, np.sqrt(0.5), size=n)
        y = W @ theta0 / np.sqrt(n) + U
        return dict(W=W, Wt=W / np.sqrt(n), y=y, n=n)

    def erm(self, d):                                   # OLS in local coordinates
        Wt = d["Wt"]
        return np.linalg.solve(Wt.T @ Wt, Wt.T @ d["y"])

    def influence(self, d, theta0):                     # theta-tilde (Lemma 2)
        u = d["y"] - d["Wt"] @ theta0                   # = U exactly here
        X = 2.0 * d["W"] * u[:, None]                   # X_i = -grad_beta l = 2 w u
        return theta0 + X.sum(0) / np.sqrt(d["n"])

    def smooth_grad(self, theta, d):
        Wt = d["Wt"]
        return 2.0 * Wt.T @ (Wt @ theta - d["y"])

    def lipschitz(self, d):
        Wt = d["Wt"]
        return 2.0 * np.linalg.eigvalsh(Wt.T @ Wt)[-1]

    def loss_point(self, theta, wt_i, y_i):
        r = y_i - wt_i @ theta
        return r * r

    def Lbar(self, theta, theta0, ctx):                 # exact: Q = 0.5 I
        d = theta - theta0
        return 0.5 * d @ d


class LogitModel:
    """Logistic regression, l(beta,z) = -log f(y|w,beta), canonical link.

    DGP chosen so that H = E[b''(0) ww'] = I and Sigma = I:
      W ~ N(0, 4 I),  Y ~ Bernoulli(sigmoid(W.theta0/sqrt(n))).
    (b''(0) = 1/4 for the logit.)  Lbar is evaluated semi-analytically on a
    fixed pool of regressors, integrating out Y.
    """
    name = "logit"
    has_closed_form_ridge = False

    def generate(self, n, theta0, rng):
        p = theta0.size
        W = rng.normal(0.0, 2.0, size=(n, p))           # N(0, 4 I)
        eta0 = W @ theta0 / np.sqrt(n)
        y = (rng.random(n) < sigmoid(eta0)).astype(float)
        return dict(W=W, Wt=W / np.sqrt(n), y=y, n=n)

    def erm(self, d, n_iter=60, ridge=1e-7):            # Newton / IRLS, tiny ridge
        Wt, y = d["Wt"], d["y"]
        theta = np.zeros(Wt.shape[1])
        eye = ridge * np.eye(Wt.shape[1])
        for _ in range(n_iter):
            eta = Wt @ theta
            pr = sigmoid(eta)
            grad = Wt.T @ (pr - y) + ridge * theta
            Wd = pr * (1.0 - pr)
            Hess = Wt.T @ (Wt * Wd[:, None]) + eye
            step = np.linalg.solve(Hess, grad)
            theta -= step
            if np.linalg.norm(step) < 1e-9:
                break
        return theta

    def influence(self, d, theta0):
        resid = d["y"] - sigmoid(d["Wt"] @ theta0)      # X_i = (y - b'(w.beta0)) w
        X = d["W"] * resid[:, None]
        return theta0 + X.sum(0) / np.sqrt(d["n"])

    def smooth_grad(self, theta, d):
        Wt = d["Wt"]
        return Wt.T @ (sigmoid(Wt @ theta) - d["y"])

    def lipschitz(self, d):
        Wt = d["Wt"]
        return 0.25 * np.linalg.eigvalsh(Wt.T @ Wt)[-1] + 1e-9

    def loss_point(self, theta, wt_i, y_i):
        eta = wt_i @ theta
        return softplus(eta) - y_i * eta

    def Lbar(self, theta, theta0, ctx):
        # Lbar_n = n * E_W[ (b(eta)-b(eta0)) - b'(eta0)(eta-eta0) ],  b = softplus.
        Wp, n = ctx["pool"], ctx["n"]
        eta = Wp @ theta / np.sqrt(n)
        eta0 = Wp @ theta0 / np.sqrt(n)
        integrand = (softplus(eta) - softplus(eta0)) - sigmoid(eta0) * (eta - eta0)
        return n * integrand.mean()


# ==========================================================================
# 4. CROSS-VALIDATION  (n-fold = leave-one-out, the object in the paper)
# ==========================================================================
def _ridge_linear_cv_and_path(d, penalty, lam_grid):
    """Closed-form LOO and full-data path for linear regression + Ridge."""
    Wt, y, n = d["Wt"], d["y"], d["n"]
    G = Wt.T @ Wt
    Ainv = np.diag(1.0 / penalty.a)
    cv = np.empty(len(lam_grid))
    thetas = []
    for k, lam in enumerate(lam_grid):
        if np.isinf(lam):
            theta = np.zeros(Wt.shape[1])
            cv[k] = float(y @ y)                        # predicts 0
        else:
            M = 2.0 * G + lam * Ainv
            Minv = np.linalg.inv(M)
            theta = Minv @ (2.0 * Wt.T @ y)
            Hdiag = 2.0 * np.einsum("ij,jk,ik->i", Wt, Minv, Wt)   # leverage
            resid = y - Wt @ theta
            cv[k] = float(np.sum((resid / (1.0 - Hdiag)) ** 2))
        thetas.append(theta)
    return cv, thetas


def _generic_cv_and_path(model, d, penalty, lam_grid, loo_max_iter=60):
    """Exact LOO by refitting, warm-started, for any model/penalty.

    Each leave-one-out problem is solved by FISTA warm-started at the full-data
    fit. Since dropping one of n points moves the solution by O(1/n), a capped
    number of warm-started iterations already reaches machine-level agreement
    with the fully converged refit (verified separately).
    """
    Wt, y, n = d["Wt"], d["y"], d["n"]
    p = Wt.shape[1]

    # full-data path (warm started along the grid, fully converged)
    full = []
    x = np.zeros(p)
    L = model.lipschitz(d)
    for lam in lam_grid:
        x = fista(lambda th: model.smooth_grad(th, d),
                  lambda v, s: penalty.prox(v, s, lam), L, x)
        full.append(x.copy())

    # leave-one-out refits, warm started from the full-data fit at each lambda
    cv = np.zeros(len(lam_grid))
    idx = np.arange(n)
    for i in range(n):
        di = dict(Wt=Wt[idx != i], y=y[idx != i], n=n)
        for k, lam in enumerate(lam_grid):
            th = fista(lambda th: model.smooth_grad(th, di),
                       lambda v, s: penalty.prox(v, s, lam), L,
                       full[k], max_iter=loo_max_iter)
            cv[k] += model.loss_point(th, Wt[i], y[i])
    return cv, full


def cv_and_path(model, d, penalty, lam_grid):
    if model.has_closed_form_ridge and isinstance(penalty, Ridge):
        return _ridge_linear_cv_and_path(d, penalty, lam_grid)
    return _generic_cv_and_path(model, d, penalty, lam_grid)


# ==========================================================================
# 5. LIMIT EXPERIMENT  (normal-means model tuned by SURE)
# ==========================================================================
def limit_experiment(penalty, theta0, lam_grid, sigma_diag, M, n_draw, rng):
    """Return (E[min(loss,M)], loss_samples) for the SURE-tuned limit.

    Vectorised over the n_draw normal draws and the lambda grid: for every
    draw we build SURE(lambda) across the grid, pick the SURE-minimising
    lambda, and record the squared error 1/2 ||theta^* - theta0||^2.
    """
    p = theta0.size
    draws = theta0 + rng.standard_normal((n_draw, p)) * np.sqrt(sigma_diag)   # (D, p)
    sure_mat = np.empty((n_draw, len(lam_grid)))
    loss_mat = np.empty((n_draw, len(lam_grid)))
    for k, lam in enumerate(lam_grid):
        g = penalty.g(draws, lam)                      # (D, p)
        dg = np.broadcast_to(penalty.dg(draws, lam), draws.shape)
        sure_mat[:, k] = 0.5 * (sigma_diag.sum() + (g * g).sum(1) + 2.0 * (dg * sigma_diag).sum(1))
        thstar = draws + g
        loss_mat[:, k] = 0.5 * ((thstar - theta0) ** 2).sum(1)
    k_star = np.argmin(sure_mat, axis=1)
    losses = np.take_along_axis(loss_mat, k_star[:, None], axis=1)[:, 0]
    return np.minimum(losses, M).mean(), losses


# ==========================================================================
# 6. ONE FINITE-SAMPLE REPLICATION  ->  the four metrics
# ==========================================================================
def one_replication(model, penalty, theta0, n, lam_grid, sigma_diag, M, ctx, rng):
    d = model.generate(n, theta0, rng)
    thetahat = model.erm(d)
    thetatil = model.influence(d, theta0)

    # --- Lemma 2: influence-function approximation of the ERM
    Delta_IF_num = np.sum((thetahat - thetatil) ** 2)
    Delta_IF_den = np.sum((thetahat - theta0) ** 2)

    # --- CV path and SURE path (the latter at the realized thetahat, true Sigma).
    #     SURE follows the manuscript definition (with the overall factor 1/2, see
    #     sure()), so it is on the same loss scale as n-fold CV and CV is compared
    #     to it directly.
    cv, full = cv_and_path(model, d, penalty, lam_grid)
    sv = np.array([sure(penalty, thetahat, lam, sigma_diag) for lam in lam_grid])

    # --- Lemma 4: recenter at lambda_ref = grid[0] to absorb the constant c_n
    cvc = cv - cv[0]
    svc = sv - sv[0]
    Delta_CV_num = np.max(np.abs(cvc - svc))
    Delta_CV_den = sv.max() - sv.min()

    # --- Lemma 5: excess loss from CV-tuning vs SURE-tuning (same thetahat)
    k_cv = int(np.argmin(cv))
    k_su = int(np.argmin(sv))
    th_cv = thetahat + penalty.g(thetahat, lam_grid[k_cv])
    th_su = thetahat + penalty.g(thetahat, lam_grid[k_su])
    loss_cv = 0.5 * np.sum((th_cv - theta0) ** 2)
    loss_su = 0.5 * np.sum((th_su - theta0) ** 2)

    # --- Theorem 1: out-of-sample loss of the fully-tuned penalized ERM
    theta_star = full[k_cv]
    Lbar_fin = model.Lbar(theta_star, theta0, ctx)

    return dict(Delta_IF_num=Delta_IF_num, Delta_IF_den=Delta_IF_den,
                Delta_CV_num=Delta_CV_num, Delta_CV_den=Delta_CV_den,
                excess=loss_cv - loss_su, sure_tuned_loss=loss_su,
                Lbar_min=min(Lbar_fin, M))


# ==========================================================================
# 7. DESIGNS
# ==========================================================================
def make_designs():
    """The five designs, all in p = 10 with round focal values of theta0.

    D1, D3 are dense (every coordinate carries the same unit signal); D2, D4 are
    sparse (three strong entries against seven exact zeros, the Lasso regime);
    D5 is the anisotropic stress test -- an isotropic-signal Ridge problem whose
    penalty matrix A has a wide (geometric, 1..40) eigenvalue spread, so that the
    SURE objective is poorly separated near its minimum for a sizeable fraction
    of draws.  (Genuine multi-modality of SURE, as in the p = 2 example of
    Figure~multimodality, is a low-dimensional knife-edge phenomenon and does not
    survive in p = 10; the flat near-optimum is the relevant high-dimensional
    stress on the well-separation argument of Lemma 5.)
    """
    p = 10
    I = np.ones(p)
    sparse = np.array([3.0, -3.0, 2.0] + [0.0] * (p - 3))   # round sparse signal
    aniso = 40.0 ** np.linspace(0.0, 1.0, p)                # geometric spread 1..40
    # The *_tex fields are the human-readable LaTeX used to auto-generate the
    # "Simulation designs" table; they are display metadata only.
    return [
        dict(key="D1", label="Linear / Ridge",
             model=LinearModel(), penalty=Ridge(I),
             theta0=np.full(p, 1.0),
             loss_tex="Linear", penalty_tex="Ridge",
             theta0_tex=r"$(1,\dots,1)$", A_tex=r"$I$"),
        dict(key="D2", label="Linear / Lasso (sparse)",
             model=LinearModel(), penalty=Lasso(I),
             theta0=sparse.copy(),
             loss_tex="Linear", penalty_tex="Lasso",
             theta0_tex=r"$(3,-3,2,0,\dots,0)$", A_tex=r"$I$"),
        dict(key="D3", label="Logit / Ridge",
             model=LogitModel(), penalty=Ridge(I),
             theta0=np.full(p, 1.0),
             loss_tex="Logit", penalty_tex="Ridge",
             theta0_tex=r"$(1,\dots,1)$", A_tex=r"$I$"),
        dict(key="D4", label="Logit / Lasso (sparse)",
             model=LogitModel(), penalty=Lasso(I),
             theta0=sparse.copy(),
             loss_tex="Logit", penalty_tex="Lasso",
             theta0_tex=r"$(3,-3,2,0,\dots,0)$", A_tex=r"$I$"),
        dict(key="D5", label="Linear / Ridge (anisotropic)",
             model=LinearModel(), penalty=Ridge(aniso),
             theta0=np.full(p, 1.5),
             loss_tex="Linear", penalty_tex="Ridge",
             theta0_tex=r"$1.5\cdot(1,\dots,1)$",
             A_tex=r"$\mathrm{diag}(40^{(j-1)/9})$"),
    ]
