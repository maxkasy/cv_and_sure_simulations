"""
Driver: run all designs, build the results table and the two figures.

Outputs (written to OUTDIR):
  results.json        raw metrics (so figures/table can be re-rendered cheaply)
  table_main.tex      LaTeX (booktabs) results table for the paper
  table_main.csv      same content as CSV
  fig1_convergence.pdf/.png
  fig2_riskfunction.pdf/.png
"""
import os, json, time, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cvsure_sim as S

# Manuscript figure directory (../Figures relative to the Simulations folder).
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "Figures"))


def _save(fig, name, outdir):
    """Save a figure as pdf+png into output/ and also into ../Figures/."""
    os.makedirs(FIGDIR, exist_ok=True)
    for d in (outdir, FIGDIR):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Configuration.  QUICK reproduces the qualitative picture in a few minutes;
# raise the rep counts (and extend the generic n-grids) for publication.
# --------------------------------------------------------------------------
CFG = dict(
    n_grid=14,
    M=20.0,
    n_report=400,                       # sample size shown in the main table
    reps_cheap=8000,                    # closed-form linear+Ridge designs (fast)
    reps_generic=2000,                  # designs needing exact LOO refits (expensive)
    n_list_cheap=[50, 100, 200, 400, 800],
    n_list_generic=[50, 100, 200, 400, 800],
    n_draw_limit=20000,                 # draws for the SURE-tuned limit
    pool_size=20000,                    # regressor pool for logit Lbar
    slice_n_list=[50, 200, 800],        # Figure 2 (risk function)
    slice_reps=1500,
    slice_t=np.round(np.linspace(0.3, 3.0, 7), 3),
    boot=400,                           # bootstrap resamples for std errors
    seed=20240629,
)


def is_cheap(des):
    return des["model"].has_closed_form_ridge and isinstance(des["penalty"], S.Ridge)


def boot_ratio(num, den, rng, B):
    """Bootstrap SE of mean(num)/mean(den) over replications."""
    num, den = np.asarray(num), np.asarray(den)
    m = len(num)
    idx = rng.integers(0, m, size=(B, m))
    vals = num[idx].mean(1) / den[idx].mean(1)
    return float(vals.std())


def boot_se(x, rng, B):
    x = np.asarray(x)
    m = len(x)
    return float(x[rng.integers(0, m, size=(B, m))].mean(1).std())


# --------------------------------------------------------------------------
# Main sweep: every (design, n) cell, collecting per-replication quantities.
# --------------------------------------------------------------------------
def run_cell(model, penalty, theta0, n, grid, sigma, M, ctx, reps, rng):
    keys = ["Delta_IF_num", "Delta_IF_den", "Delta_CV_num", "Delta_CV_den",
            "excess", "sure_tuned_loss", "Lbar_min"]
    acc = {k: np.empty(reps) for k in keys}
    for r in range(reps):
        out = S.one_replication(model, penalty, theta0, n, grid, sigma, M, ctx, rng)
        for k in keys:
            acc[k][r] = out[k]
    return acc


def main(outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    OUTDIR = outdir
    t_start = time.time()
    rng = np.random.default_rng(CFG["seed"])
    designs = S.make_designs()
    pool_full = rng.normal(0.0, 2.0, size=(CFG["pool_size"], 10))

    records = []          # one row per (design, n)
    limitR = {}           # design key -> limiting risk
    for des in designs:
        key, model, pen, th0 = des["key"], des["model"], des["penalty"], des["theta0"]
        p = th0.size
        grid = pen.lam_grid(CFG["n_grid"])
        sigma = np.ones(p)
        n_list = CFG["n_list_cheap"] if is_cheap(des) else CFG["n_list_generic"]
        reps = CFG["reps_cheap"] if is_cheap(des) else CFG["reps_generic"]

        Rlim, _ = S.limit_experiment(pen, th0, grid, sigma, CFG["M"],
                                     CFG["n_draw_limit"], rng)
        limitR[key] = Rlim

        for n in n_list:
            ctx = dict(pool=pool_full[:, :p], n=n)
            t0 = time.time()
            acc = run_cell(model, pen, th0, n, grid, sigma, CFG["M"], ctx, reps, rng)
            Delta_IF = acc["Delta_IF_num"].mean() / acc["Delta_IF_den"].mean()
            Delta_CV = acc["Delta_CV_num"].mean() / acc["Delta_CV_den"].mean()
            Delta_tune = np.abs(acc["excess"]).mean() / acc["sure_tuned_loss"].mean()
            Rn = acc["Lbar_min"].mean()
            # Diagnostic columns use the manuscript notation:
            #   Delta_IF  (Lemma 2),   Delta_CV (Lemma 4),   Delta_tune (Lemma 5),
            #   R_n / R  (Theorem 1, limit),   Delta_R = |R_n - R|.
            records.append(dict(
                key=key, label=des["label"], model=model.name,
                penalty=pen.name, p=p, n=n, reps=reps,
                Delta_IF=Delta_IF, Delta_IF_se=boot_ratio(acc["Delta_IF_num"], acc["Delta_IF_den"], rng, CFG["boot"]),
                Delta_CV=Delta_CV, Delta_CV_se=boot_ratio(acc["Delta_CV_num"], acc["Delta_CV_den"], rng, CFG["boot"]),
                Delta_tune=Delta_tune, Delta_tune_se=boot_ratio(acc["excess"], acc["sure_tuned_loss"], rng, CFG["boot"]),
                R_n=Rn, R_n_se=boot_se(acc["Lbar_min"], rng, CFG["boot"]),
                R=Rlim, Delta_R=abs(Rn - Rlim),
            ))
            print(f"[{time.time()-t_start:6.1f}s] {key} n={n:4d} reps={reps}: "
                  f"dIF={Delta_IF:.3f} dCV={Delta_CV:.3f} dtune={Delta_tune:+.3f} "
                  f"R_n={Rn:.3f} R={Rlim:.3f} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(records)

    # ---------------------------------------------------------------- Figure 2
    # Risk function R_n(theta0) along a slice theta0 = t * v, design D1.
    d1 = next(d for d in designs if d["key"] == "D1")
    model, pen = d1["model"], d1["penalty"]
    p = d1["theta0"].size
    v = np.ones(p) / np.sqrt(p)
    grid = pen.lam_grid(CFG["n_grid"])
    sigma = np.ones(p)
    slice_curves = {n: [] for n in CFG["slice_n_list"]}
    slice_limit = []
    for t in CFG["slice_t"]:
        th0 = t * v
        Rl, _ = S.limit_experiment(pen, th0, grid, sigma, CFG["M"],
                                   CFG["n_draw_limit"], rng)
        slice_limit.append(Rl)
        for n in CFG["slice_n_list"]:
            vals = np.empty(CFG["slice_reps"])
            for r in range(CFG["slice_reps"]):
                d = model.generate(n, th0, rng)
                thh = model.erm(d)
                cv, full = S.cv_and_path(model, d, pen, grid)
                k = int(np.argmin(cv))
                vals[r] = min(model.Lbar(full[k], th0, None), CFG["M"])
            slice_curves[n].append(vals.mean())
        print(f"[{time.time()-t_start:6.1f}s] slice t={t}: done", flush=True)

    results = dict(
        config={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in CFG.items()},
        table=df.to_dict(orient="records"),
        limitR=limitR,
        slice=dict(t=CFG["slice_t"].tolist(), limit=slice_limit,
                   curves={str(n): slice_curves[n] for n in CFG["slice_n_list"]}),
    )
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    render_designs_table(designs, OUTDIR)
    render_table(df, OUTDIR)
    render_fig1(df, designs, OUTDIR)
    render_fig2(results["slice"], OUTDIR)
    print(f"[{time.time()-t_start:6.1f}s] DONE", flush=True)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_designs_table(designs, outdir):
    """Write the 'Simulation designs' table as a complete, labelled table float."""
    p = designs[0]["theta0"].size
    rows = []
    for d in designs:
        rows.append(
            f"{d['key']} & {d['loss_tex']} & {d['penalty_tex']} & "
            f"{d['theta0'].size} & {d['theta0_tex']} & {d['A_tex']} \\\\")
    # Short caption above the table; the detailed note goes below the tabular
    # (\flushleft), and the whole float is set \footnotesize.
    note = (
        r"$\Sigma=I$ and $p=" + str(p) + r"$ throughout (normalized coordinates). "
        r"The sparse signal is $\theta_0=(3,-3,2,0,\dots,0)$; for D5, "
        r"$A=\mathrm{diag}(40^{0},40^{1/9},\dots,40^{1})$ is a geometric grid "
        r"spanning $[1,40]$.")
    latex = "\n".join([
        r"% Auto-generated by run_all.render_designs_table -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Simulation designs. }",
        r"\label{tab:sim_designs}",
        r"\vspace{4pt}",
        r"",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"& Loss & Penalty & $p$ & $\theta_0$ & $A$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\flushleft",
        note,
        r"\end{table}",
    ])
    with open(os.path.join(outdir, "table_designs.tex"), "w") as f:
        f.write(latex + "\n")


def render_table(df, outdir):
    """Write the diagnostics table (at n = CFG['n_report']) as a complete,
    labelled table float, plus a CSV with the same content."""
    n_rep = CFG["n_report"]
    sub = df[df["n"] == n_rep].copy().sort_values("key")
    cols = ["label", "p", "n", "Delta_IF", "Delta_CV", "Delta_tune", "R_n", "R", "Delta_R"]
    sub[cols].to_csv(os.path.join(outdir, "table_main.csv"), index=False)

    reps_cf = int(sub["reps"].max())          # closed-form designs (D1, D5)
    reps_loo = int(sub["reps"].min())         # leave-one-out designs (D2--D4)
    body = []
    for _, r in sub.iterrows():
        name = f"{r['key']}\\quad {r['label']}" if "key" in r else str(r["label"])
        body.append(
            f"{name} & {r['Delta_IF']:.3f} & {r['Delta_CV']:.3f} "
            f"& {r['Delta_tune']:.3f} & {r['R_n']:.3f} & {r['R']:.3f} & {r['Delta_R']:.3f} \\\\")
    # Short caption above the table; the detailed column note goes below the
    # tabular (\flushleft), and the whole float is set \footnotesize.
    caption = r"Approximation diagnostics at $n=" + str(n_rep) + r"$. "
    note = (
        r"Columns: $\Delta_{\mathrm{IF}}$ (Lemma~\ref{lem:ifrep}), "
        r"$\Delta_{\mathrm{CV}}$ (Lemma~\ref{lem:convergence_cv}), "
        r"$\Delta_{\mathrm{tune}}$ (Lemma~\ref{lem:convergencetuned}), the "
        r"finite-sample and limiting risks $R_n,R$, and their gap $\Delta_{R}$ "
        r"(Theorem~\ref{theo:risk_convergence}). Averages over $" + str(reps_cf) +
        r"$ replications for the closed-form designs D1, D5 and $" + str(reps_loo) +
        r"$ for the leave-one-out designs D2--D4.")
    latex = "\n".join([
        r"% Auto-generated by run_all.render_table -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{" + caption + r"}",
        r"\label{tab:sim_results}",
        r"\vspace{4pt}",
        r"",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Design & $\Delta_{\mathrm{IF}}$ & $\Delta_{\mathrm{CV}}$ & "
        r"$\Delta_{\mathrm{tune}}$ & $R_n$ & $R$ & $\Delta_{R}$ \\",
        r"& (Lem.~2) & (Lem.~4) & (Lem.~5) & (Thm.~1) & (limit) & $|R_n-R|$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\flushleft",
        note,
        r"\end{table}",
    ])
    with open(os.path.join(outdir, "table_main.tex"), "w") as f:
        f.write(latex + "\n")


def _logticks(ax, vals):
    """Label a log axis at exactly the given values (no ambiguous bare decades)."""
    ax.yaxis.set_major_locator(mticker.FixedLocator(vals))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def render_fig1(df, designs, outdir):
    """Convergence figure: a 2x2 layout with three data panels (the two vanishing
    approximation errors and the risk convergence) plus a dedicated legend panel,
    so the legend never overlaps the curves."""
    from matplotlib.lines import Line2D
    FS = dict(title=18, label=17, tick=15, legend=16, slope=16, suptitle=22)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(designs)))
    nticks = [50, 100, 200, 400, 800]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.0), constrained_layout=True)

    # Top row: the two approximation errors that vanish (log-log).
    panelspec = [
        (axes[0, 0], "Delta_IF", r"$\Delta_{\mathrm{IF}}$",
         r"$\Delta_{\mathrm{IF}}$ (Lemma 2: influence-function approx.)",
         -1.0, r"$\propto n^{-1}$", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]),
        (axes[0, 1], "Delta_CV", r"$\Delta_{\mathrm{CV}}$",
         r"$\Delta_{\mathrm{CV}}$ (Lemma 4: CV $\approx$ SURE)",
         -0.5, r"$\propto n^{-1/2}$", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0])]
    for ax, m, ylab, title, slope, slope_lbl, yt in panelspec:
        for des, c in zip(designs, colors):
            sub = df[df["key"] == des["key"]].sort_values("n")
            y = np.where(sub[m].to_numpy() <= 0, np.nan, sub[m].to_numpy())
            ax.plot(sub["n"], y, "o-", color=c, lw=2.4, ms=7, label=des["key"])
        nn = np.array([50.0, 800.0]); base = df[df["key"] == "D1"].sort_values("n")[m].iloc[0]
        ax.plot(nn, base * (nn / 50.0) ** slope, "k--", lw=1.5, alpha=0.6)
        ax.text(170, base * (170 / 50.0) ** slope * 1.5, slope_lbl, fontsize=FS["slope"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(nticks); ax.set_xticklabels([str(v) for v in nticks])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        _logticks(ax, yt)
        ax.set_xlabel("sample size $n$", fontsize=FS["label"])
        ax.set_ylabel(ylab, fontsize=FS["label"])
        ax.set_title(title, fontsize=FS["title"])
        ax.tick_params(axis="both", labelsize=FS["tick"])
        ax.grid(True, which="both", ls=":", alpha=0.4)

    # Bottom-left: finite-sample risk R_n converging to the limit R (Theorem 1).
    axc = axes[1, 0]
    for des, c in zip(designs, colors):
        sub = df[df["key"] == des["key"]].sort_values("n")
        axc.errorbar(sub["n"], sub["R_n"], yerr=sub["R_n_se"], fmt="o-", color=c,
                     lw=2.4, ms=7, capsize=3)
        axc.axhline(sub["R"].iloc[0], color=c, ls=":", lw=1.6, alpha=0.8)
    axc.set_xscale("log")
    axc.set_xticks(nticks); axc.set_xticklabels([str(v) for v in nticks])
    axc.xaxis.set_minor_locator(mticker.NullLocator())
    axc.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    axc.set_xlabel("sample size $n$", fontsize=FS["label"])
    axc.set_ylabel(r"risk  $E[\min(\bar L_n,\,M)]$", fontsize=FS["label"])
    axc.set_title(r"$R_n \to R$ (Theorem 1; dotted = limit $R$)", fontsize=FS["title"])
    axc.tick_params(axis="both", labelsize=FS["tick"])
    axc.grid(True, which="both", ls=":", alpha=0.4)

    # Bottom-right: dedicated legend panel (keeps the legend off the curves).
    lax = axes[1, 1]
    lax.axis("off")
    handles = [Line2D([0], [0], color=c, marker="o", lw=2.4, ms=8) for c in colors]
    labels = [f"{des['key']}: {des['label']}" for des in designs]
    handles.append(Line2D([0], [0], color="0.4", ls=":", lw=2.0))
    labels.append(r"dotted: limiting risk $R$")
    lax.legend(handles, labels, loc="center", fontsize=FS["legend"],
               title="Designs", title_fontsize=FS["legend"], frameon=True,
               framealpha=0.95, handlelength=2.4, borderpad=1.0, labelspacing=0.9)

    fig.suptitle("Convergence of the key approximations as $n$ grows", fontsize=FS["suptitle"])
    _save(fig, "fig1_convergence", outdir)


def render_fig2(slice_data, outdir):
    t = np.array(slice_data["t"])
    fig, ax = plt.subplots(figsize=(6.4, 4.7), constrained_layout=True)
    ns = sorted(int(k) for k in slice_data["curves"])
    blues = plt.cm.Blues(np.linspace(0.45, 0.9, len(ns)))
    for n, c in zip(ns, blues):
        ax.plot(t, slice_data["curves"][str(n)], "o-", color=c, lw=2.0, ms=6,
                label=f"$R_n$, $n={n}$")
    ax.plot(t, slice_data["limit"], "k--", lw=2.2, label="limit $R$ (SURE)")
    ax.set_xlabel(r"signal strength $t$  ($\theta_0 = t\,v$,  $\|\theta_0\|=t$)", fontsize=13)
    ax.set_ylabel(r"risk  $E[\min(\bar L_n,\,M)]$", fontsize=13)
    ax.set_xticks(t); ax.set_xticklabels([f"{x:g}" for x in t])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Convergence of the risk function (Linear / Ridge)", fontsize=14)
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=12)
    _save(fig, "fig2_riskfunction", outdir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
