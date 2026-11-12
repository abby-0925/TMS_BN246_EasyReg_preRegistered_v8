TMS 队列 BN246：EasyReg-MNI + SynthStrip + 病灶感知 R2SN v8
============================================================

目标
----
对 Data_flip 中全部 TMS T1 统一使用 FreeSurfer 8.0.0 mri_easyreg 配准至 FSL MNI152 1mm；
对已在 MNI 空间且已左右翻转的病灶 mask，只做最近邻重采样到 EasyReg T1 的实际网格，
不再次左右翻转、不再次 mri_easywarp；然后 SynthStrip 去颅骨，按 BN246 提取 R2SN。

重要原则
--------
1. T1 来自 Data_flip，保持现有左右翻转状态；脚本不再翻转 T1。
2. 病灶 mask 来自 lesion_mask_MNI1mm_flip，假定已在 MNI 且已完成左右翻转。
3. 病灶 mask 只通过 nibabel resample_from_to(order=0) 对齐到 EasyReg T1 网格，不做二次标准化。
4. 找不到病灶文件的被试按用户既定规则进入 NO_LESION：使用去颅骨后的 T1 和完整 BN246（但仍与脑掩膜相交）。
   注意：无病灶文件不自动等于健康对照；正式统计前应与临床分组表核对。
5. R2SN 使用相同 BN246 1mm 图谱、R2SN Params.yaml、picked_features.txt、22 个选定特征、
   每个特征跨有效 ROI Min-Max，再对区域特征向量计算 Pearson 相似性。
6. 病灶排除 margin=0 mm，与 sub-TMS002ses01 成功试跑保持一致。
7. 新结果写入 TMS_BN246_preRegistered_v8，不覆盖 v7 试跑。

服务器路径（按本项目已确认位置）
--------------------------------
Raw T1:
  /data02/linzhenzong/TMS-MRI/Data_flip/sub-TMSxxxsesxx/anat/sub-TMSxxxsesxx_T1w.nii.gz

Already-flipped MNI lesion:
  /data02/linzhenzong/TMS-MRI/Lesion/lesion_mask_MNI1mm_flip/
  filename:
  sub-TMSxxxsesxx_space-MNI152NLin6Asym_res-01_label-lesion_mask.nii.gz

FSL MNI152 1mm:
  /data01/software/fsl/data/standard/MNI152_T1_1mm.nii.gz

BN246 1mm:
  /data02/tanghui/C3_R2SN/Atlases/BN_Atlas_246_1mm.nii.gz

FreeSurfer:
  /usr/local/freesurfer/8.0.0

Output root:
  /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/

安装/环境
---------
conda activate r2sn
export FREESURFER_HOME=/usr/local/freesurfer/8.0.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
hash -r

检查：
  which mri_easyreg
  which mri_synthstrip
  python -c 'import R2SN, ants, nibabel, pandas, numpy, radiomics; print("R2SN environment OK")'

将本目录所有 .py 文件放在同一个 scripts 文件夹中。

阶段 0：生成 manifest
--------------------
python make_manifest.py

输出：
  /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/manifest_TMS_BN246.csv

检查：
  column id / t1 / lesion / lesion_mode
  MNI_MASK = 找到翻转后的 MNI lesion 文件
  NO_LESION = 没找到匹配的 lesion 文件
先人工核对 NO_LESION 名单与临床表，尤其是卒中患者病灶文件是否缺失。

阶段 1：所有 T1 EasyReg-MNI + SynthStrip
----------------------------------------
先小规模试跑（建议至少包含一名 MNI_MASK 和一名 NO_LESION）：
  python run_preprocess_parallel.py --workers 2 --threads-per-worker 2 --limit 3

确认：
  pre_registered/<ID>/T1_EasyReg_MNI.nii.gz
  pre_registered/<ID>/T1_EasyReg_MNI_brain.nii.gz
  pre_registered/<ID>/T1_EasyReg_MNI_brainmask.nii.gz
  pre_registered/<ID>/lesion_MNI_on_T1grid.nii.gz
  pre_registered/<ID>/T1prep_lesion_overlay.png
  pre_registered/<ID>/preprocess_summary.json

确认配准和 lesion overlay 正常后，跑全队列：
  nohup python -u run_preprocess_parallel.py \
    --workers 4 --threads-per-worker 2 \
    > /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/preprocess_all.log 2>&1 < /dev/null &

监控：
  tail -f /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/preprocess_all.log
  python check_status.py

阶段 2：所有预处理成功者进行 R2SN
--------------------------------
先试跑：
  python run_r2sn_parallel.py --workers 2 --threads-per-worker 2 --limit 3 --keep-qc-images

检查：
  subjects/<ID>/TMS_BN246_preRegistered_v8/qc/run_summary.json
  subjects/<ID>/TMS_BN246_preRegistered_v8/qc/T1prep_lesion_overlay.png
  subjects/<ID>/TMS_BN246_preRegistered_v8/radiomics/ROI_features_246.csv
  subjects/<ID>/TMS_BN246_preRegistered_v8/network/<ID>_network.npy

确认 pilot 成功后，全队列：
  nohup python -u run_r2sn_parallel.py \
    --workers 5 --threads-per-worker 2 \
    --keep-qc-images \
    > /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/R2SN_all_v8.log 2>&1 < /dev/null &

监控：
  tail -f /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/R2SN_all_v8.log
  python check_status.py

并行建议
--------
EasyReg 阶段先 4 workers × 2 threads。
R2SN 阶段先 5 workers × 2 threads。
不要同时启动两个全队列阶段；必须等阶段 1 全部结束并检查 QC 后再启动阶段 2。
如果服务器负载或内存持续偏高，降低 workers 至 3。

输出目录
--------
pre_registered/<ID>/
  T1_EasyReg_MNI.nii.gz
  T1_EasyReg_MNI_brain.nii.gz
  T1_EasyReg_MNI_brainmask.nii.gz
  lesion_MNI_on_T1grid.nii.gz
  T1prep_lesion_overlay.png
  preprocess_summary.json
  preprocess.log

subjects/<ID>/TMS_BN246_preRegistered_v8/
  qc/run_summary.json
  qc/ROI_lesion_overlap.csv
  qc/BN_lesionExcluded.nii.gz
  qc/registered_T1_MNI.nii.gz (使用 --keep-qc-images)
  qc/lesion_resampled_MNI.nii.gz (使用 --keep-qc-images)
  radiomics/ROI_features_246.csv
  radiomics/ROI_node_metrics.csv
  radiomics/pyradiomics_input_radiomics_features.csv
  network/<ID>_network.npy
  network/<ID>_valid_roi_mask.npy
  network/<ID>_network_zeroFilled_DISPLAY_ONLY.npy

状态解释
--------
SUCCESS: 246 个 ROI 中所有非空节点均成功提取。
SUCCESS_WITH_MISSING_NODES: 有非空 ROI 的特征提取失败或无效；需检查 missing_rois。
FAILED: 处理失败，查看对应日志和 run_summary.json。
FULLY_LESIONED_ZERO: ROI 被病灶完全覆盖，网络对应行列编码 0。
FEATURE_EXTRACTION_FAILED / INVALID_FEATURE_VALUES: 非空 ROI 特征提取失败，网络对应行列为 NaN，不应当作病灶零节点。

注意
----
- 不能只凭输出文件名含 MNI 就认定配准正确，必须抽查 T1prep_lesion_overlay.png。
- 不要把 network_zeroFilled_DISPLAY_ONLY.npy 用于正式统计；正式分析使用 network.npy + valid_roi_mask.npy。
- 该包按已知目录/命名模式编写。首次运行务必先小规模试跑，查看 preprocess.log 中 mri_easyreg 命令是否正常，
  并人工确认至少一名病灶患者、一名无病灶文件患者的 QC。
