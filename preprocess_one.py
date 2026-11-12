#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One subject: FreeSurfer EasyReg to FSL MNI152 1mm, SynthStrip, align MNI lesion."""
import argparse, json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

def run_cmd(cmd, log, env):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        p = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}); see {log}: {' '.join(map(str, cmd))}")

def same_grid(a, b, tol=0.02):
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and np.allclose(a.affine, b.affine, atol=tol, rtol=0)

def save_like(data, ref, out, dtype=np.uint8):
    h = ref.header.copy()
    h.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), ref.affine, h)
    img.set_qform(ref.affine, code=1)
    img.set_sform(ref.affine, code=1)
    nib.save(img, str(out))

def make_qc(t1_path, brainmask_path, lesion_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t1 = nib.load(str(t1_path))
    a = np.asarray(t1.dataobj, dtype=np.float32)
    bm = np.asarray(nib.load(str(brainmask_path)).dataobj) > 0
    les = np.asarray(nib.load(str(lesion_path)).dataobj) > 0
    coords = np.argwhere(les)
    if len(coords):
        cx, cy, cz = np.median(coords, axis=0).astype(int)
    else:
        coords = np.argwhere(bm)
        cx, cy, cz = np.median(coords, axis=0).astype(int)
    cx = int(np.clip(cx, 0, a.shape[0]-1))
    cy = int(np.clip(cy, 0, a.shape[1]-1))
    cz = int(np.clip(cz, 0, a.shape[2]-1))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    views = [
        (np.rot90(a[:, :, cz]), np.rot90(les[:, :, cz]), "Axial"),
        (np.rot90(a[:, cy, :]), np.rot90(les[:, cy, :]), "Coronal"),
        (np.rot90(a[cx, :, :]), np.rot90(les[cx, :, :]), "Sagittal"),
    ]
    for x, (im, mask, title) in zip(ax, views):
        x.imshow(im, cmap="gray", origin="lower")
        if mask.any():
            overlay = np.ma.masked_where(~mask, mask)
            x.imshow(overlay, cmap="Reds", alpha=0.65, origin="lower", vmin=0, vmax=1)
        x.set_title(title + (" | lesion red" if les.any() else " | no lesion"))
        x.axis("off")
    fig.suptitle(Path(t1_path).parent.name + ": EasyReg-MNI T1 / SynthStrip / lesion-grid QC")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--t1", required=True)
    p.add_argument("--lesion", default="NONE")
    p.add_argument("--lesion-mode", choices=["MNI_MASK", "NO_LESION"], required=True)
    p.add_argument("--mni", default="/data01/software/fsl/data/standard/MNI152_T1_1mm.nii.gz")
    p.add_argument("--out-root", default="/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/pre_registered")
    p.add_argument("--freesurfer-home", default="/usr/local/freesurfer/8.0.0")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--keep-transforms", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    sid = a.id
    out = Path(a.out_root) / sid
    out.mkdir(parents=True, exist_ok=True)
    log = out / "preprocess.log"
    summary = out / "preprocess_summary.json"
    outputs = [
        out / "T1_EasyReg_MNI.nii.gz",
        out / "T1_EasyReg_MNI_brain.nii.gz",
        out / "T1_EasyReg_MNI_brainmask.nii.gz",
        out / "lesion_MNI_on_T1grid.nii.gz",
        out / "T1prep_lesion_overlay.png",
    ]
    if not a.force and summary.is_file() and all(x.is_file() and x.stat().st_size > 0 for x in outputs):
        try:
            old = json.loads(summary.read_text())
            if old.get("status") == "SUCCESS":
                print(f"[{sid}] SKIP_PREPROCESS_SUCCESS")
                return
        except Exception:
            pass

    fs = Path(a.freesurfer_home)
    easyreg = fs / "bin" / "mri_easyreg"
    synthstrip = fs / "bin" / "mri_synthstrip"
    for exe in (easyreg, synthstrip):
        if not exe.is_file():
            raise SystemExit(f"FreeSurfer executable not found: {exe}")
    for f in (Path(a.t1), Path(a.mni)):
        if not f.is_file():
            raise SystemExit(f"Input not found: {f}")
    if a.lesion_mode == "MNI_MASK" and not Path(a.lesion).is_file():
        raise SystemExit(f"lesion_mode=MNI_MASK but lesion file missing: {a.lesion}")

    env = os.environ.copy()
    env["FREESURFER_HOME"] = str(fs)
    env["PATH"] = str(fs / "bin") + os.pathsep + env.get("PATH", "")
    for k in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
              "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(a.threads)

    ref_seg = out / "MNI152_ref_synthseg.nii.gz"
    flo_seg = out / "T1_native_synthseg.nii.gz"
    fwd = out / "EasyReg_forward.nii.gz"
    bak = out / "EasyReg_backward.nii.gz"
    status = {"id": sid, "t1": a.t1, "lesion": a.lesion, "lesion_mode": a.lesion_mode,
              "mni": a.mni, "threads": a.threads, "status": "STARTED",
              "time_start": time.strftime("%F %T")}
    try:
        if a.force:
            for x in outputs + [ref_seg, flo_seg, fwd, bak]:
                if x.exists():
                    x.unlink()

        # EasyReg will create SynthSeg segmentations at --ref_seg / --flo_seg if absent.
        cmd = [
            str(easyreg), "--ref", a.mni, "--ref_seg", str(ref_seg),
            "--flo", a.t1, "--flo_seg", str(flo_seg),
            "--flo_reg", str(outputs[0]),
            "--fwd_field", str(fwd), "--bak_field", str(bak),
            "--threads", str(a.threads),
        ]
        run_cmd(cmd, log, env)
        if not outputs[0].is_file():
            raise RuntimeError("EasyReg did not create T1_EasyReg_MNI.nii.gz")

        cmd = [str(synthstrip), "-i", str(outputs[0]), "-o", str(outputs[1]),
               "-m", str(outputs[2]), "--threads", str(a.threads)]
        run_cmd(cmd, log, env)

        mni_img = nib.load(a.mni)
        t1_img = nib.load(str(outputs[0]))
        brain_img = nib.load(str(outputs[1]))
        bm_img = nib.load(str(outputs[2]))
        if not same_grid(t1_img, mni_img):
            raise RuntimeError(f"EasyReg output is not on the exact MNI reference grid: "
                               f"T1={t1_img.shape}/{t1_img.affine.tolist()} "
                               f"MNI={mni_img.shape}/{mni_img.affine.tolist()}")
        if not same_grid(brain_img, t1_img) or not same_grid(bm_img, t1_img):
            raise RuntimeError("SynthStrip outputs do not share the EasyReg T1 grid")

        bm = np.asarray(bm_img.dataobj) > 0
        if int(bm.sum()) < 100000:
            raise RuntimeError(f"SynthStrip brain mask suspiciously small: {int(bm.sum())} voxels")

        if a.lesion_mode == "MNI_MASK":
            les_src = nib.load(a.lesion)
            if len(les_src.shape) != 3:
                raise RuntimeError("Lesion mask must be 3D")
            if same_grid(les_src, t1_img):
                les_img = les_src
                resampled = False
            else:
                les_img = resample_from_to(les_src, t1_img, order=0, mode="constant", cval=0)
                resampled = True
            les = np.asarray(les_img.dataobj)
            if not np.isfinite(les).all():
                raise RuntimeError("Lesion mask contains NaN/Inf")
            les = les > 0
            if not les.any():
                raise RuntimeError("MNI lesion became empty after nearest-neighbor resampling; check orientation/affine")
            overlap = float((les & bm).sum() / max(int(les.sum()), 1))
            if overlap < 0.05:
                raise RuntimeError(f"Only {overlap:.3f} of lesion voxels overlap SynthStrip brain mask; inspect alignment")
        else:
            les = np.zeros(t1_img.shape[:3], dtype=bool)
            resampled = False
            overlap = None

        save_like(les, t1_img, outputs[3], np.uint8)
        make_qc(outputs[1], outputs[2], outputs[3], outputs[4])

        status.update({
            "status": "SUCCESS",
            "T1_shape": list(t1_img.shape[:3]),
            "T1_voxel_mm": list(nib.affines.voxel_sizes(t1_img.affine)),
            "brainmask_voxels": int(bm.sum()),
            "lesion_voxels": int(les.sum()),
            "lesion_resampled_to_T1_grid": bool(resampled),
            "lesion_brain_overlap_fraction": overlap,
            "outputs": [str(x) for x in outputs],
        })
        if not a.keep_transforms:
            for x in (ref_seg, flo_seg, fwd, bak):
                if x.exists():
                    x.unlink()
    except Exception as e:
        status.update({"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})
        raise
    finally:
        status["time_end"] = time.strftime("%F %T")
        tmp = summary.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(summary)
        print(f"[{sid}] {status['status']}: {status.get('error', '')}", flush=True)

if __name__ == "__main__":
    main()
