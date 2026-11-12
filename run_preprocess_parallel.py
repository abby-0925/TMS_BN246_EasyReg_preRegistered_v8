#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patient-level parallel EasyReg + SynthStrip preprocessing (no Slurm)."""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"id", "t1", "lesion", "lesion_mode"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in manifest")
    return rows

def run_one(row, a, logdir):
    sid = row["id"]
    out = Path(a.out_root) / sid
    summary = out / "preprocess_summary.json"
    expected = [out/"T1_EasyReg_MNI.nii.gz", out/"T1_EasyReg_MNI_brain.nii.gz",
                out/"T1_EasyReg_MNI_brainmask.nii.gz", out/"lesion_MNI_on_T1grid.nii.gz",
                out/"T1prep_lesion_overlay.png"]
    if not a.force and summary.is_file() and all(x.is_file() and x.stat().st_size for x in expected):
        try:
            if json.loads(summary.read_text()).get("status") == "SUCCESS":
                return {"id":sid, "state":"SKIP_SUCCESS", "seconds":0, "log":""}
        except Exception:
            pass
    log = logdir / f"{sid}.log"
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("preprocess_one.py")),
           "--id", sid, "--t1", row["t1"], "--lesion", row["lesion"],
           "--lesion-mode", row["lesion_mode"], "--mni", a.mni,
           "--out-root", a.out_root, "--freesurfer-home", a.freesurfer_home,
           "--threads", str(a.threads_per_worker)]
    if a.keep_transforms: cmd.append("--keep-transforms")
    if a.force: cmd.append("--force")
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS","ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
              "OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[k] = str(a.threads_per_worker)
    t0 = time.monotonic()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%F %T')} START {sid} =====\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    state = "SUCCESS" if rc == 0 else "FAILED"
    return {"id":sid, "state":state, "returncode":rc,
            "seconds":round(time.monotonic()-t0,1), "log":str(log)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/manifest_TMS_BN246.csv")
    p.add_argument("--mni", default="/data01/software/fsl/data/standard/MNI152_T1_1mm.nii.gz")
    p.add_argument("--out-root", default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/pre_registered")
    p.add_argument("--freesurfer-home", default="/usr/local/freesurfer/8.0.0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--id", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-transforms", action="store_true")
    a=p.parse_args()
    rows=read_manifest(a.manifest)
    if a.id: rows=[r for r in rows if r["id"]==a.id]
    else:
        rows=rows[a.start:]
        if a.limit: rows=rows[:a.limit]
    if not rows: raise SystemExit("No subjects selected")
    logdir=Path(a.out_root).parent/"logs_preprocess"
    logdir.mkdir(parents=True, exist_ok=True)
    prog=Path(a.out_root).parent/"preprocess_progress.csv"
    print(f"EasyReg preprocessing: n={len(rows)} workers={a.workers} threads/worker={a.threads_per_worker}", flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(run_one,r,a,logdir):r["id"] for r in rows}
        for i,f in enumerate(as_completed(futures),1):
            try: res=f.result()
            except Exception as e: res={"id":futures[f],"state":"LAUNCHER_FAILED","detail":str(e)}
            results.append(res)
            print(f"[{i}/{len(rows)}] {res['id']} {res['state']} {res.get('seconds','')}s", flush=True)
            with prog.open("w",newline="",encoding="utf-8-sig") as g:
                w=csv.DictWriter(g,fieldnames=sorted({k for x in results for k in x}))
                w.writeheader();w.writerows(results)
    print("COUNTS", dict(Counter(x["state"] for x in results)))
    print("PROGRESS", prog)

if __name__=="__main__":
    main()
