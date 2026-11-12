#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel R2SN stage for subjects already preprocessed to MNI by EasyReg."""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

def read_manifest(path):
    with open(path,newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    required={"id","code","lesion_mode"}
    if not rows or not required.issubset(rows[0]): raise ValueError(f"Manifest needs {required}")
    if len({r["id"] for r in rows})!=len(rows): raise ValueError("Duplicate IDs")
    return rows

def run_one(row,a,logdir):
    sid=row["id"]
    prep=Path(a.prep_root)/sid
    required=[prep/"T1_EasyReg_MNI_brain.nii.gz",prep/"T1_EasyReg_MNI_brainmask.nii.gz",
              prep/"lesion_MNI_on_T1grid.nii.gz",prep/"preprocess_summary.json"]
    if not all(x.is_file() for x in required):
        return {"id":sid,"state":"PREPROCESS_MISSING","detail":str(prep)}
    try:
        if json.loads(required[-1].read_text()).get("status")!="SUCCESS":
            return {"id":sid,"state":"PREPROCESS_NOT_SUCCESS","detail":str(required[-1])}
    except Exception:
        return {"id":sid,"state":"PREPROCESS_SUMMARY_INVALID"}
    base=Path(a.out_root)/sid/a.tag
    summary=base/"qc"/"run_summary.json"
    if not a.force and summary.is_file():
        try:
            old=json.loads(summary.read_text())
            if old.get("status") in ("SUCCESS","SUCCESS_WITH_MISSING_NODES") and \
               (base/"network"/f"{sid}_network.npy").is_file() and \
               (base/"radiomics"/"ROI_features_246.csv").is_file():
                return {"id":sid,"state":"SKIP_SUCCESS","seconds":0}
        except Exception: pass
    log=logdir/f"{sid}.log"
    cmd=[sys.executable,"-u",str(Path(__file__).with_name("run_one_preregistered.py")),
         "--id",sid,"--code",row.get("code",sid),"--prep-dir",str(prep),
         "--lesion-mode",row["lesion_mode"],"--out-root",a.out_root,"--tag",a.tag,
         "--tmp-root",a.tmp_root,"--margin-mm",str(a.margin_mm),
         "--min-voxels",str(a.min_voxels)]
    if a.keep_qc_images: cmd.append("--keep-qc-images")
    if a.force: cmd.append("--force")
    env=os.environ.copy()
    for k in ("OMP_NUM_THREADS","ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
              "OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[k]=str(a.threads_per_worker)
    t0=time.monotonic()
    with log.open("a",encoding="utf-8") as f:
        f.write(f"\n===== START {time.strftime('%F %T')} {sid} =====\n")
        f.write(" ".join(cmd)+"\n"); f.flush()
        rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    state="SUCCESS" if rc==0 else "FAILED"
    if summary.is_file():
        try: state=json.loads(summary.read_text()).get("status",state)
        except Exception: pass
    return {"id":sid,"state":state,"returncode":rc,
            "seconds":round(time.monotonic()-t0,1),"log":str(log)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/manifest_TMS_BN246.csv")
    p.add_argument("--prep-root",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/pre_registered")
    p.add_argument("--out-root",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects")
    p.add_argument("--tag",default="TMS_BN246_preRegistered_v8")
    p.add_argument("--tmp-root",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/tmp")
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--threads-per-worker",type=int,default=2)
    p.add_argument("--margin-mm",type=float,default=0.0)
    p.add_argument("--min-voxels",type=int,default=200)
    p.add_argument("--limit",type=int,default=0); p.add_argument("--start",type=int,default=0)
    p.add_argument("--id",default=""); p.add_argument("--force",action="store_true")
    p.add_argument("--keep-qc-images",action="store_true")
    a=p.parse_args()
    rows=read_manifest(a.manifest)
    if a.id: rows=[r for r in rows if r["id"]==a.id]
    else:
        rows=rows[a.start:]
        if a.limit: rows=rows[:a.limit]
    if not rows: raise SystemExit("No subjects selected")
    logdir=Path(a.out_root).parent/"logs_r2sn"; logdir.mkdir(parents=True,exist_ok=True)
    prog=Path(a.out_root).parent/f"{a.tag}_progress.csv"
    print(f"R2SN preRegistered: n={len(rows)} workers={a.workers} threads/worker={a.threads_per_worker}",flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(run_one,r,a,logdir):r["id"] for r in rows}
        for i,f in enumerate(as_completed(fs),1):
            try: res=f.result()
            except Exception as e: res={"id":fs[f],"state":"LAUNCHER_FAILED","detail":str(e)}
            results.append(res)
            print(f"[{i}/{len(rows)}] {res['id']} {res['state']} {res.get('seconds','')}s",flush=True)
            with prog.open("w",newline="",encoding="utf-8-sig") as g:
                w=csv.DictWriter(g,fieldnames=sorted({k for x in results for k in x}))
                w.writeheader();w.writerows(results)
    print("COUNTS",dict(Counter(x["state"] for x in results)))
    print("PROGRESS",prog)

if __name__=="__main__": main()
