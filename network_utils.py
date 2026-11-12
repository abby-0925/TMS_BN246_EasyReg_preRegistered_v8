#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2SN BN246 network construction helpers."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

NROI=246

def classify_rois(atlas, lesion, excluded_mask, nroi=NROI, min_voxels=200):
    if atlas.shape != lesion.shape or atlas.shape != excluded_mask.shape:
        raise ValueError("atlas/lesion/excluded_mask shape mismatch")
    atlas_mask=(atlas>=1)&(atlas<=nroi)
    labels=atlas[atlas_mask].astype(np.int32,copy=False)
    total=np.bincount(labels,minlength=nroi+1)
    lesion_n=np.bincount(labels[lesion[atlas_mask]],minlength=nroi+1)
    excl_n=np.bincount(labels[excluded_mask[atlas_mask]],minlength=nroi+1)
    clean=np.where(atlas_mask&(~excluded_mask),atlas,0).astype(np.int16)
    rows=[]
    for i in range(1,nroi+1):
        nt=int(total[i]); nl=int(lesion_n[i]); ne=int(excl_n[i]); left=nt-ne
        frac=nl/nt if nt else float("nan")
        if nt==0: st="ATLAS_MISSING"
        elif nl==nt: st="FULLY_LESIONED_ZERO"
        elif left==0: st="EMPTY_AFTER_MARGIN_ZERO"
        elif left<min_voxels: st="LOW_VOXELS_ATTEMPT"
        else: st="EXTRACT_PENDING"
        rows.append(dict(ROI=i,total_voxels=nt,lesion_voxels=nl,lesion_fraction=frac,
                         excluded_voxels=ne,clean_voxels=left,
                         clean_fraction=left/nt if nt else float("nan"),status=st))
    return clean,rows

def valid_feature_network(feature_df,picked,qc_rows,nroi=NROI,eps=1e-12):
    if len(qc_rows)!=nroi or [int(r["ROI"]) for r in qc_rows]!=list(range(1,nroi+1)):
        raise ValueError("QC rows must contain ROI labels 1..246 in order")
    for k in ["Label"]+list(picked):
        if len(feature_df) and k not in feature_df.columns:
            raise ValueError("Missing feature column: "+k)
    if len(feature_df):
        labels=pd.to_numeric(feature_df["Label"],errors="coerce")
        if labels.isna().any(): raise ValueError("Non-numeric feature label")
        labels=labels.astype(int)
        if labels.duplicated().any() or not labels.between(1,nroi).all():
            raise ValueError("Duplicate/out-of-range feature labels")
        feature_df=feature_df.copy()
        feature_df["Label"]=labels
        feature_df=feature_df.set_index("Label")
    else:
        feature_df=pd.DataFrame(index=pd.Index([],name="Label"))
    Xfull=np.full((nroi,len(picked)),np.nan,dtype=float)
    states=[]
    zero_states={"FULLY_LESIONED_ZERO","EMPTY_AFTER_MARGIN_ZERO"}
    for idx,r in enumerate(qc_rows):
        old=r["status"]
        if old in zero_states or old=="ATLAS_MISSING":
            states.append(old); continue
        label=idx+1
        if label not in feature_df.index:
            states.append("FEATURE_EXTRACTION_FAILED"); continue
        vals=pd.to_numeric(feature_df.loc[label,list(picked)],errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            states.append("INVALID_FEATURE_VALUES"); continue
        Xfull[idx]=vals
        states.append("LOW_VOXELS_EXTRACTED" if old=="LOW_VOXELS_ATTEMPT" else "VALID")
    candidate=np.array([i for i,s in enumerate(states) if s in ("VALID","LOW_VOXELS_EXTRACTED")],dtype=int)
    W=np.full((nroi,nroi),np.nan,dtype=float)
    zidx=[i for i,s in enumerate(states) if s in zero_states]
    if zidx:
        W[zidx,:]=0.0; W[:,zidx]=0.0
    used=[]; dropped=[]
    if len(candidate):
        X=Xfull[candidate]
        spread=X.max(axis=0)-X.min(axis=0)
        keep=np.isfinite(spread)&(spread>eps)
        used=[p for p,k in zip(picked,keep) if k]
        dropped=[p for p,k in zip(picked,keep) if not k]
        if int(keep.sum())>=2:
            X=(X[:,keep]-X[:,keep].min(axis=0))/(X[:,keep].max(axis=0)-X[:,keep].min(axis=0))
            C=X-X.mean(axis=1,keepdims=True)
            norms=np.linalg.norm(C,axis=1)
            good=norms>eps
            for j in candidate[~good]: states[j]="DEGENERATE_NORMALIZED_VECTOR"
            valid=candidate[good]
            if len(valid):
                U=C[good]/norms[good,None]
                sub=np.clip(U@U.T,-1,1)
                sub=(sub+sub.T)/2
                np.fill_diagonal(sub,1.0)
                W[np.ix_(valid,valid)]=sub
        else:
            for j in candidate: states[j]="INSUFFICIENT_VARIABLE_FEATURES"
    valid_mask=np.array([s in ("VALID","LOW_VOXELS_EXTRACTED") for s in states],dtype=bool)
    details=dict(n_valid=int(valid_mask.sum()),n_zero=len(zidx),
                 n_other_missing=int(nroi-valid_mask.sum()-len(zidx)),
                 n_low_voxels_extracted=sum(s=="LOW_VOXELS_EXTRACTED" for s in states),
                 selected_features=len(picked),used_features=len(used),
                 dropped_constant_features=dropped,zero_rois=[i+1 for i in zidx],
                 missing_rois=[i+1 for i,s in enumerate(states) if not valid_mask[i] and i not in zidx])
    return W,valid_mask,Xfull,states,details

def nodal_metrics(W,valid_mask,qc_rows,statuses):
    rows=[]
    for i in range(W.shape[0]):
        a=W[i].copy(); a[i]=np.nan
        vals=a[np.isfinite(a)]
        active=a[valid_mask & np.isfinite(a)]
        rows.append(dict(ROI=i+1,node_status=statuses[i],
            lesion_fraction=qc_rows[i]["lesion_fraction"],clean_voxels=qc_rows[i]["clean_voxels"],
            n_defined_edges=int(np.isfinite(a).sum()),n_active_edges=int(len(active)),
            mean_similarity_all_defined=float(vals.mean()) if len(vals) else float("nan"),
            mean_similarity_active=float(active.mean()) if len(active) else float("nan"),
            positive_strength_all_defined=float(np.clip(vals,0,None).sum()) if len(vals) else float("nan"),
            negative_abs_strength_all_defined=float(np.clip(-vals,0,None).sum()) if len(vals) else float("nan"),
            is_zero_coded=int(statuses[i] in ("FULLY_LESIONED_ZERO","EMPTY_AFTER_MARGIN_ZERO"))))
    return rows

def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def atomic_save_npy(path,arr):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f: np.save(f,arr)
    tmp.replace(path)
