"""
Chunked driver (so each invocation finishes quickly and results accumulate
on disk).  Subcommands:

  python chunk.py cell KEY N REPS     run REPS more reps of design KEY at size N
  python chunk.py limit               limiting SURE-tuned risk for every design
  python chunk.py slice REPS          risk-function slice for Figure 2 (design D1)
  python chunk.py render              build results.json, table and both figures

Cells accumulate into output/cell_<KEY>_<N>.npz, so a slow design can be run in
several passes.  All randomness is seeded reproducibly from CFG['seed'].
"""
import os, sys, json, glob, time
import numpy as np
import cvsure_sim as S
from run_all import (CFG, is_cheap, boot_ratio, boot_se,
                     render_designs_table, render_table, render_fig1, render_fig2)

OUTDIR = os.environ.get("CVSURE_OUTDIR", "output")
os.makedirs(OUTDIR, exist_ok=True)
# Raw per-replication accumulators stored in each cell's npz, from which the
# manuscript diagnostics (Delta_IF, Delta_CV, Delta_tune, R_n, R, Delta_R) are
# derived in cmd_render. The num/den pairs carry the diagnostic they form; the
# tuning and risk diagnostics keep their descriptive component names.
KEYS = ["Delta_IF_num", "Delta_IF_den", "Delta_CV_num", "Delta_CV_den",
        "excess", "sure_tuned_loss", "Lbar_min"]


def stable_id(key):                              # process-independent seed offset
    return sum(ord(c) for c in key)


def design_by_key(key):
    return next(d for d in S.make_designs() if d["key"] == key)


def pool_for(p):
    rng = np.random.default_rng([CFG["seed"], 999])      # fixed pool
    return rng.normal(0.0, 2.0, size=(CFG["pool_size"], p))


def cell_path(key, n):
    return os.path.join(OUTDIR, f"cell_{key}_{n}.npz")


def cmd_cell(key, n, reps):
    des = design_by_key(key)
    model, pen, th0 = des["model"], des["penalty"], des["theta0"]
    p = th0.size
    grid = pen.lam_grid(CFG["n_grid"])
    sigma = np.ones(p)
    ctx = dict(pool=pool_for(p), n=n)

    path = cell_path(key, n)
    have = {k: [] for k in KEYS}
    batch = 0
    if os.path.exists(path):
        z = np.load(path)
        have = {k: list(z[k]) for k in KEYS}
        batch = int(z["batch"])
    # fresh, reproducible draws for this batch
    rng = np.random.default_rng([CFG["seed"], stable_id(key), n, batch])

    t0 = time.time()
    for r in range(reps):
        out = S.one_replication(model, pen, th0, n, grid, sigma, CFG["M"], ctx, rng)
        for k in KEYS:
            have[k].append(out[k])
    np.savez(path, batch=batch + 1, **{k: np.asarray(have[k]) for k in KEYS})
    tot = len(have["Lbar_min"])
    print(f"{key} n={n}: +{reps} reps (total {tot}) in {time.time()-t0:.1f}s", flush=True)


def cmd_limit():
    out = {}
    for des in S.make_designs():
        rng = np.random.default_rng([CFG["seed"], 7, hash(des["key"]) % (2**31)])
        grid = des["penalty"].lam_grid(CFG["n_grid"])
        sigma = np.ones(des["theta0"].size)
        R, _ = S.limit_experiment(des["penalty"], des["theta0"], grid, sigma,
                                  CFG["M"], CFG["n_draw_limit"], rng)
        out[des["key"]] = R
        print(f"limit {des['key']}: R={R:.4f}", flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "limits.json"), "w"), indent=2)


def cmd_slice(reps):
    rng = np.random.default_rng([CFG["seed"], 13])
    d1 = design_by_key("D1")
    model, pen = d1["model"], d1["penalty"]
    p = d1["theta0"].size
    v = np.ones(p) / np.sqrt(p)
    grid = pen.lam_grid(CFG["n_grid"])
    sigma = np.ones(p)
    curves = {n: [] for n in CFG["slice_n_list"]}
    limit = []
    for t in CFG["slice_t"]:
        th0 = t * v
        R, _ = S.limit_experiment(pen, th0, grid, sigma, CFG["M"],
                                  CFG["n_draw_limit"], rng)
        limit.append(R)
        for n in CFG["slice_n_list"]:
            vals = np.empty(reps)
            for r in range(reps):
                d = model.generate(n, th0, rng)
                cv, full = S.cv_and_path(model, d, pen, grid)
                k = int(np.argmin(cv))
                vals[r] = min(model.Lbar(full[k], th0, None), CFG["M"])
            curves[n].append(vals.mean())
        print(f"slice t={t}: done", flush=True)
    json.dump(dict(t=list(map(float, CFG["slice_t"])), limit=limit,
                   curves={str(n): curves[n] for n in CFG["slice_n_list"]}),
              open(os.path.join(OUTDIR, "slice.json"), "w"), indent=2)


def cmd_render():
    import pandas as pd
    limits = json.load(open(os.path.join(OUTDIR, "limits.json")))
    rng = np.random.default_rng([CFG["seed"], 4242])
    rows = []
    for des in S.make_designs():
        key = des["key"]
        for path in sorted(glob.glob(cell_path(key, 0).replace("_0.npz", "_*.npz"))):
            z = np.load(path)
            n = int(path.split("_")[-1].split(".")[0])
            a = {k: z[k] for k in KEYS}
            Rn = a["Lbar_min"].mean()
            # Diagnostic columns use the manuscript notation (Delta_IF, Delta_CV,
            # Delta_tune, R_n, R, Delta_R), formed from the raw per-rep accumulators.
            rows.append(dict(
                key=key, label=des["label"], model=des["model"].name,
                penalty=des["penalty"].name, p=des["theta0"].size, n=n,
                reps=len(a["Lbar_min"]),
                Delta_IF=a["Delta_IF_num"].mean() / a["Delta_IF_den"].mean(),
                Delta_IF_se=boot_ratio(a["Delta_IF_num"], a["Delta_IF_den"], rng, CFG["boot"]),
                Delta_CV=a["Delta_CV_num"].mean() / a["Delta_CV_den"].mean(),
                Delta_CV_se=boot_ratio(a["Delta_CV_num"], a["Delta_CV_den"], rng, CFG["boot"]),
                Delta_tune=np.abs(a["excess"]).mean() / a["sure_tuned_loss"].mean(),
                Delta_tune_se=boot_ratio(np.abs(a["excess"]), a["sure_tuned_loss"], rng, CFG["boot"]),
                R_n=Rn, R_n_se=boot_se(a["Lbar_min"], rng, CFG["boot"]),
                R=limits[key], Delta_R=abs(Rn - limits[key])))
    df = pd.DataFrame(rows).sort_values(["key", "n"])
    df.to_csv(os.path.join(OUTDIR, "metrics_long.csv"), index=False)

    slice_data = json.load(open(os.path.join(OUTDIR, "slice.json")))
    json.dump(dict(config={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in CFG.items()},
                   table=df.to_dict(orient="records"),
                   limitR=limits, slice=slice_data),
              open(os.path.join(OUTDIR, "results.json"), "w"), indent=2)

    designs = S.make_designs()
    render_designs_table(designs, OUTDIR)
    render_table(df, OUTDIR)
    render_fig1(df, designs, OUTDIR)
    render_fig2(slice_data, OUTDIR)
    print("rendered tables + figures from", len(df), "cells", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cell":
        cmd_cell(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "limit":
        cmd_limit()
    elif cmd == "slice":
        cmd_slice(int(sys.argv[2]))
    elif cmd == "render":
        cmd_render()
    else:
        raise SystemExit(f"unknown command {cmd}")
