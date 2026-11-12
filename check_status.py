#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json
from pathlib import Path
from collections import Counter

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/manifest_TMS_BN246.csv")
    p.add_argument("--prep-root",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/pre_registered")
    p.add_argument("--out-root",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects")
    p.add_argument("--tag",default="TMS_BN246_preRegistered_v8")
    a=p.parse_args()
    with open(a.manifest,newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    prep=Counter(); r2=Counter(); bad=[]
    for row in rows:
        sid=row["id"]
        ps=Path(a.prep_root)/sid/"preprocess_summary.json"
        if ps.is_file():
            try: st=json.loads(ps.read_text()).get("status","UNKNOWN")
            except Exception: st="BAD_JSON"
        else: st="NOT_STARTED"
        prep[st]+=1
        rs=Path(a.out_root)/sid/a.tag/"qc"/"run_summary.json"
        if rs.is_file():
            try:
                m=json.loads(rs.read_text()); st2=m.get("status","UNKNOWN")
                if st2 not in ("SUCCESS","SUCCESS_WITH_MISSING_NODES"):
                    bad.append((sid,st2,m.get("error","")))
            except Exception: st2="BAD_JSON"; bad.append((sid,st2,""))
        else: st2="NOT_STARTED"
        r2[st2]+=1
    print("N_SUBJECTS",len(rows))
    print("PREPROCESS",dict(prep))
    print("R2SN",dict(r2))
    if bad:
        print("\nR2SN failures:")
        for x in bad: print(*x,sep=" | ")

if __name__=="__main__": main()
