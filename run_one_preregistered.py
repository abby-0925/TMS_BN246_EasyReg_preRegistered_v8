#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2SN from pre-registered EasyReg-MNI T1/brainmask/lesion; NO second registration."""
import argparse, ast, csv, json, os, shutil, tempfile, time, traceback, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from network_utils import classify_rois, valid_feature_network, nodal_metrics, write_rows, atomic_save_npy

VERSION="TMS_BN246_preRegistered_v8"

def same_grid(a,b,tol=0.02):
    return tuple(a.shape[:3])==tuple(b.shape[:3]) and np.allclose(a.affine,b.affine,atol=tol,rtol=0)

def save_like(data,ref,path,dtype=np.uint8):
    h=ref.header.copy(); h.set_data_dtype(dtype)
    img=nib.Nifti1Image(np.asarray(data,dtype=dtype),ref.affine,h)
    img.set_qform(ref.affine,code=1); img.set_sform(ref.affine,code=1)
    nib.save(img,str(path))

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def package_defaults():
    import R2SN
    root=Path(R2SN.__file__).resolve().parent
    return {"atlas":Path("/data02/tanghui/C3_R2SN/Atlases/BN_Atlas_246_1mm.nii.gz"),
            "target":Path("/data01/software/fsl/data/standard/MNI152_T1_1mm.nii.gz"),
            "params":root/"network_feature"/"Params.yaml",
            "picked":root/"network_feature"/"picked_features.txt"}

def n4_normalize(brain_path,mask_path,out_path):
    import ants
    im=ants.image_read(str(brain_path))
    mk=ants.image_read(str(mask_path))
    corrected=ants.n4_bias_field_correction(im,mask=mk,shrink_factor=2,
        convergence={"iters":[50,50,50,50],"tol":1e-7})
    arr=np.asarray(corrected.numpy(),dtype=np.float32)
    mask=np.asarray(mk.numpy())>0
    vals=arr[mask & np.isfinite(arr)]
    if vals.size<100000: raise RuntimeError(f"N4 brain mask too small: {vals.size}")
    p95=float(np.percentile(vals,95))
    if not np.isfinite(p95) or p95<=0: raise RuntimeError(f"Invalid N4 95th percentile: {p95}")
    arr=np.clip(arr/p95*255.0,0,255).astype(np.float32)
    ref=nib.load(str(brain_path))
    nib.save(nib.Nifti1Image(arr,ref.affine,ref.header),str(out_path))
    return p95

def dilate_mm(mask,affine,mm):
    if mm<=0 or not mask.any(): return mask.copy()
    return distance_transform_edt(~mask,sampling=nib.affines.voxel_sizes(affine)[:3])<=mm

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--id",required=True); p.add_argument("--code",default="")
    p.add_argument("--prep-dir",required=True)
    p.add_argument("--lesion-mode",choices=["MNI_MASK","NO_LESION"],required=True)
    p.add_argument("--out-root",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects")
    p.add_argument("--tag",default="TMS_BN246_preRegistered_v8")
    p.add_argument("--atlas"); p.add_argument("--target"); p.add_argument("--params"); p.add_argument("--picked")
    p.add_argument("--tmp-root",default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/tmp")
    p.add_argument("--margin-mm",type=float,default=0.0)
    p.add_argument("--min-voxels",type=int,default=200)
    p.add_argument("--keep-qc-images",action="store_true")
    p.add_argument("--force",action="store_true")
    a=p.parse_args()

    import R2SN
    from R2SN.network_feature.feature_extract import feature_extract
    defaults=package_defaults()
    cfg={k:Path(getattr(a,k) or defaults[k]) for k in defaults}
    for k,v in cfg.items():
        if not v.is_file(): raise FileNotFoundError(f"{k}: {v}")
    prep=Path(a.prep_dir)
    t1=prep/"T1_EasyReg_MNI_brain.nii.gz"
    bm=prep/"T1_EasyReg_MNI_brainmask.nii.gz"
    lesp=prep/"lesion_MNI_on_T1grid.nii.gz"
    for x in (t1,bm,lesp):
        if not x.is_file(): raise FileNotFoundError(x)

    base=Path(a.out_root)/a.id/a.tag
    qc,rad,nd=base/"qc",base/"radiomics",base/"network"
    for d in (qc,rad,nd): d.mkdir(parents=True,exist_ok=True)
    summary=qc/"run_summary.json"
    meta={"version":VERSION,"id":a.id,"code":a.code,"prep_dir":str(prep),
          "lesion_mode":a.lesion_mode,"tag":a.tag,"nroi":246,
          "margin_mm":a.margin_mm,"min_voxels":a.min_voxels,
          "atlas":str(cfg["atlas"]),"target":str(cfg["target"]),
          "params_sha256":sha256(cfg["params"]),"picked_sha256":sha256(cfg["picked"]),
          "normalization":"N4 then p95 brain-intensity scaling to 0-255",
          "feature_normalization":"per-feature min-max across valid surviving ROIs",
          "network":"Pearson correlation across selected R2SN radiomic features"}
    if not a.force and summary.is_file():
        try:
            old=json.loads(summary.read_text())
            expected=["SUCCESS","SUCCESS_WITH_MISSING_NODES"]
            if old.get("status") in expected and all((nd/f"{a.id}_network.npy").is_file() and
                (rad/"ROI_features_246.csv").is_file() and (nd/f"{a.id}_valid_roi_mask.npy").is_file()):
                if all(str(old.get(k))==str(v) for k,v in meta.items()):
                    print(f"[{a.id}] SKIP_EXISTING_SUCCESS"); return
        except Exception: pass

    meta["status"]="STARTED"; meta["time_start"]=time.strftime("%F %T")
    try:
        target=nib.load(str(cfg["target"]))
        atlas_img=nib.load(str(cfg["atlas"]))
        t1_img=nib.load(str(t1)); bm_img=nib.load(str(bm)); les_img=nib.load(str(lesp))
        for name,img in [("atlas",atlas_img),("registered T1",t1_img),("brainmask",bm_img),("lesion",les_img)]:
            if not same_grid(img,target):
                raise RuntimeError(f"{name} not on exact target MNI grid")
        atlasf=np.asarray(atlas_img.dataobj)
        atlas=np.rint(atlasf).astype(np.int16)
        if any(not np.any(atlas==i) for i in range(1,247)):
            raise RuntimeError("BN246 atlas does not contain every label 1..246")
        brainmask=np.asarray(bm_img.dataobj)>0
        if int(brainmask.sum())<100000: raise RuntimeError("brain mask is suspiciously small")
        if a.lesion_mode=="MNI_MASK":
            lesion=np.asarray(les_img.dataobj)>0
            if not lesion.any(): raise RuntimeError("MNI_MASK mode but lesion is empty")
            overlap=float((lesion&brainmask).sum()/max(int(lesion.sum()),1))
            if overlap<0.05: raise RuntimeError(f"lesion/brain overlap too small ({overlap:.3f}); inspect alignment")
        else:
            lesion=np.zeros(target.shape[:3],dtype=bool)
            overlap=None
        # Atlas is already in MNI152 1mm; intersect with each subject's brain mask.
        atlas_in_brain=np.where(brainmask & (atlas>=1) & (atlas<=246),atlas,0).astype(np.int16)
        expanded=dilate_mm(lesion,target.affine,a.margin_mm)
        clean,rows=classify_rois(atlas_in_brain,lesion,expanded,246,a.min_voxels)
        meta.update({"brainmask_voxels":int(brainmask.sum()),"lesion_voxels":int(lesion.sum()),
                     "lesion_volume_mm3":float(lesion.sum()*abs(np.linalg.det(target.affine[:3,:3]))),
                     "lesion_brain_overlap_fraction":overlap,
                     "n_atlas_voxels_in_brain":int((atlas_in_brain>0).sum()),
                     "n_lesioned_rois":sum(r["lesion_voxels"]>0 for r in rows),
                     "n_fully_lesioned_rois":sum(r["status"]=="FULLY_LESIONED_ZERO" for r in rows),
                     "n_atlas_missing_rois":sum(r["status"]=="ATLAS_MISSING" for r in rows)})
        write_rows(qc/"ROI_lesion_overlap.csv",rows)
        clean_path=qc/"BN_lesionExcluded.nii.gz"
        save_like(clean,target,clean_path,np.int16)
        if a.keep_qc_images:
            shutil.copy2(t1,qc/"registered_T1_MNI_brain.nii.gz")
            shutil.copy2(bm,qc/"patient_brain_mask_MNI.nii.gz")
            shutil.copy2(lesp,qc/"lesion_resampled_MNI.nii.gz")
            shutil.copy2(prep/"T1_EasyReg_MNI.nii.gz",qc/"registered_T1_MNI.nii.gz")
            shutil.copy2(prep/"T1prep_lesion_overlay.png",qc/"T1prep_lesion_overlay.png")

        tmp_root=Path(a.tmp_root); tmp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"R2SN_preReg_{a.id}_",dir=tmp_root) as td:
            td=Path(td)
            norm=td/"T1_R2SN_input_MNI.nii.gz"
            p95=n4_normalize(t1,bm,norm)
            meta["n4_p95_before_scaling"]=p95
            csv_in=td/"pyradiomics_input.csv"
            candidates=[r["ROI"] for r in rows if r["clean_voxels"]>0]
            with csv_in.open("w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["Image","Mask","Label"])
                for roi in candidates: w.writerow([str(norm),str(clean_path),roi])
            rtemp=td/"radiomics"; rtemp.mkdir()
            if candidates:
                feature_name=feature_extract(str(csv_in),str(cfg["params"]),str(rtemp))
                feature_csv=rtemp/feature_name
                feat=pd.read_csv(feature_csv) if feature_csv.is_file() else pd.DataFrame()
            else:
                feature_csv=None; feat=pd.DataFrame()
            picked=ast.literal_eval(cfg["picked"].read_text(encoding="utf-8"))
            W,valid,full,states,stats=valid_feature_network(feat,picked,rows,246)
            for row,st in zip(rows,states): row["status"]=st
            write_rows(qc/"ROI_lesion_overlap.csv",rows)
            fdf=pd.DataFrame(full,columns=picked)
            fdf.insert(0,"status",states); fdf.insert(0,"ROI",range(1,247))
            fdf.to_csv(rad/"ROI_features_246.csv",index=False)
            if feature_csv and feature_csv.is_file():
                shutil.copy2(feature_csv,rad/"pyradiomics_input_radiomics_features.csv")
            write_rows(rad/"ROI_node_metrics.csv",nodal_metrics(W,valid,rows,states))
            atomic_save_npy(nd/f"{a.id}_network.npy",W)
            atomic_save_npy(nd/f"{a.id}_valid_roi_mask.npy",valid)
            atomic_save_npy(nd/f"{a.id}_network_zeroFilled_DISPLAY_ONLY.npy",np.nan_to_num(W,nan=0.0))
            meta.update(stats)
            meta["feature_rows_requested"]=len(candidates)
            meta["feature_rows_extracted"]=len(feat)
            meta["network_path"]=str(nd/f"{a.id}_network.npy")
            meta["radiomics_path"]=str(rad/"ROI_features_246.csv")
            meta["status"]="SUCCESS_WITH_MISSING_NODES" if stats["n_other_missing"] else "SUCCESS"
        return
    except Exception as e:
        meta.update({"status":"FAILED","error":f"{type(e).__name__}: {e}",
                     "traceback":traceback.format_exc()})
        raise
    finally:
        meta["time_end"]=time.strftime("%F %T")
        tmp=summary.with_name(summary.name+".tmp")
        tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        tmp.replace(summary)
        print(f"[{a.id}] {meta['status']} valid={meta.get('n_valid',0)} zero={meta.get('n_zero',0)} missing={meta.get('n_other_missing',0)} {meta.get('error','')}",flush=True)

if __name__=="__main__":
    main()
