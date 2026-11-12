#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover all TMS T1s and pair already-left-right-flipped MNI lesion masks."""
import argparse, csv
from pathlib import Path

DEFAULT_T1_ROOT = "/data02/linzhenzong/TMS-MRI/Data_flip"
DEFAULT_LESION_ROOT = "/data02/linzhenzong/TMS-MRI/Lesion/lesion_mask_MNI1mm_flip"
DEFAULT_OUT = "/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/manifest_TMS_BN246.csv"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t1-root", default=DEFAULT_T1_ROOT)
    p.add_argument("--lesion-root", default=DEFAULT_LESION_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()
    t1root, lesionroot = Path(a.t1_root), Path(a.lesion_root)
    if not t1root.is_dir():
        raise SystemExit(f"T1 root not found: {t1root}")
    if not lesionroot.is_dir():
        raise SystemExit(f"Lesion root not found: {lesionroot}")

    t1s = sorted(t1root.glob("sub-*/anat/*_T1w.nii.gz"))
    if not t1s:
        raise SystemExit(f"No T1 files found under {t1root}/sub-*/anat/*_T1w.nii.gz")

    rows, seen = [], set()
    ambiguous = []
    for t1 in t1s:
        sid = t1.parent.parent.name
        if sid in seen:
            ambiguous.append((sid, str(t1)))
            continue
        seen.add(sid)

        exact = lesionroot / f"{sid}_space-MNI152NLin6Asym_res-01_label-lesion_mask.nii.gz"
        if exact.is_file():
            lesion = exact
        else:
            matches = sorted(lesionroot.glob(f"{sid}*label-lesion_mask.nii.gz"))
            if len(matches) == 1:
                lesion = matches[0]
            elif len(matches) > 1:
                ambiguous.append((sid, "MULTIPLE_LESIONS:" + ";".join(map(str, matches))))
                continue
            else:
                lesion = None

        rows.append({
            "id": sid,
            "code": sid,
            "t1": str(t1.resolve()),
            "lesion": str(lesion.resolve()) if lesion else "NONE",
            "lesion_mode": "MNI_MASK" if lesion else "NO_LESION",
        })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "code", "t1", "lesion", "lesion_mode"])
        w.writeheader()
        w.writerows(rows)

    n_lesion = sum(r["lesion_mode"] == "MNI_MASK" for r in rows)
    n_none = len(rows) - n_lesion
    print(f"T1 subjects: {len(rows)}")
    print(f"MNI lesion masks found: {n_lesion}")
    print(f"No lesion file -> NO_LESION: {n_none}")
    print(f"Manifest: {out}")
    if ambiguous:
        print("\nWARNING: duplicate/ambiguous records skipped:")
        for x in ambiguous:
            print(x)
    print("\nIMPORTANT: NO_LESION means no matching lesion file was found; it does not independently prove the participant is a healthy control.")

if __name__ == "__main__":
    main()
