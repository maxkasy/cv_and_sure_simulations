"""
Parallel driver -- the recommended way to reproduce everything.

Runs every (design, n) cell in its own process across the available cores, then
computes the limiting risks, the Figure-2 slice, and renders both tables and both
figures. Each cell is one reproducible batch (chunk.cmd_cell with a fixed seed
keyed by design/n), so the run is deterministic and can be reproduced or extended.

    python run_parallel.py                 # full run, all cores
    python run_parallel.py --workers 8     # cap the number of worker processes

Replication counts come from run_all.CFG (reps_cheap / reps_generic). Cells
accumulate into output/cell_<KEY>_<N>.npz, identically to chunk.py, so a run can
also be extended one cell at a time with `python chunk.py cell <KEY> <N> <REPS>`.
"""
import os, sys, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import cvsure_sim as S
import chunk as C
from run_all import CFG, is_cheap


def _cell(args):
    key, n, reps = args
    C.cmd_cell(key, n, reps)
    return key, n, reps


def main(workers):
    designs = S.make_designs()
    jobs = []
    for des in designs:
        cheap = is_cheap(des)
        reps = CFG["reps_cheap"] if cheap else CFG["reps_generic"]
        nlist = CFG["n_list_cheap"] if cheap else CFG["n_list_generic"]
        for n in nlist:
            jobs.append((des["key"], n, reps))
    # heaviest cells first (generic + large n) for better load balancing
    order = {d["key"]: i for i, d in enumerate(designs)}
    jobs.sort(key=lambda j: (is_cheap(designs[order[j[0]]]), -j[1]))

    t0 = time.time()
    print(f"launching {len(jobs)} cells on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_cell, j): j for j in jobs}
        for f in as_completed(futs):
            key, n, reps = f.result()
            print(f"[{time.time()-t0:8.1f}s] cell done  {key} n={n} reps={reps}",
                  flush=True)

    print(f"[{time.time()-t0:8.1f}s] limits ...", flush=True)
    C.cmd_limit()
    print(f"[{time.time()-t0:8.1f}s] slice ...", flush=True)
    C.cmd_slice(CFG["slice_reps"])
    print(f"[{time.time()-t0:8.1f}s] render ...", flush=True)
    C.cmd_render()
    print(f"[{time.time()-t0:8.1f}s] ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 4))
    a = ap.parse_args()
    main(a.workers)
