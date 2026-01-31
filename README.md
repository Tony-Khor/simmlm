# SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality

The repo contains the codebase used for our main task (**BraTS 2018**) in the paper.  
The codebase can also be configured to run on the **LLD-MMRI** liver tumor dataset structure described below.

---

## Data Preparation

### 1. nnUNet-preprocessed BraTS 2018 Training Set
As described in our paper, we preprocess the raw MRI scans using the nnUNet preprocessing pipeline.  
You may either:

- Download the raw BraTS 2018 data **[here](https://www.kaggle.com/datasets/sanglequang/brats2018)** and run nnUNet preprocessing yourself using the official nnUNet repository: **[nnUNet GitHub Repo](https://github.com/MIC-DKFZ/nnUNet)**.  
- Download our preprocessed, reay-to-use dataset: **[Preprocessed BraTS 2018 Data](https://drive.google.com/file/d/1aCu15hDd4k0ea2MQP_5wGDXasIWlx0Iu/view?usp=sharing)**.

Place the preprocessed samples here:

```
assets/data/nnUNet_preprocessed_BraTS2018/
```

---

### LLD-MMRI (8 MRI modalities + liver tumor labels)

Expected folder layout:

```
assets/data/lld-mmri/
  images/
    <CASE_ID>_C+A_0000.nii.gz
    <CASE_ID>_C+Delay_0000.nii.gz
    <CASE_ID>_C+V_0000.nii.gz
    <CASE_ID>_C-pre_0000.nii.gz
    <CASE_ID>_DWI_0000.nii.gz
    <CASE_ID>_InPhase_0000.nii.gz
    <CASE_ID>_OutPhase_0000.nii.gz
    <CASE_ID>_T2WI_0000.nii.gz
  labels/
    <CASE_ID>_C+A.nii.gz
    <CASE_ID>_C+Delay.nii.gz
    <CASE_ID>_C+V.nii.gz
    <CASE_ID>_C-pre.nii.gz
    <CASE_ID>_DWI.nii.gz
    <CASE_ID>_InPhase.nii.gz
    <CASE_ID>_OutPhase.nii.gz
    <CASE_ID>_T2WI.nii.gz
```

By default, the pipeline reads all eight modalities and uses the `C+A` label file as the segmentation target.
You can change the modality order, label modality, and train/val/test split ratios in
`configs_expert_pretraining.py` and `configs_joint_training.py`.

To preprocess and standardize the dataset into `.npy` files, run:

```
python dataset/preprocess_lld_mmri.py \
  --input-dir assets/data/lld-mmri \
  --output-dir assets/data/lld-mmri_preprocessed \
  --modalities C+A C+Delay C+V C-pre DWI InPhase OutPhase T2WI \
  --label-modality C+A
```

Then set `USE_PREPROCESSED = True` and `SPLITS_FILE_PATH = "assets/data/lld-mmri_preprocessed/splits.json"` in the config files.

---

### 2. Training Set Split File

We provide the split file used in all our experiments:

```
assets/kfold_splits.json
```

The raw training set is divided into five 80/20 train/eval folds.  

In our published experiments, we only used fold 0.  
Note that, although we only trained with the fold-0 split, the main results reported in our paper were obtained by submitting predictions on the BraTS 2018 Official Validation Set (whose labels were not public; also see the next part) to the official evaluation platform.

---

### 3. BraTS 2018 Official Validation Set

The official validation set (without labels) can be downloaded from **[this link](https://www.kaggle.com/datasets/sanglequang/brats2018?select=MICCAI_BraTS_2018_Data_Validation)**.

Place all validation samples in:

```
assets/data/BraTS2018_eval/
```

However, since the **[BraTS 2018 official evaluation server](https://www.cbica.upenn.edu/ipp/)** is no longer available, it is currently impossible to compute official metrics for this set.  
Thus, the code for running predictions on this set has been disabled in our release by default.

---

## Environment Setup

SimMLM has minimal dependencies.  
If you use `conda` for environment management, simply run:

```
bash environment_setup.sh
```

This script will automatically create a working environment named **simmlm** with all required packages installed.

---

## Running Stage 1: Independent Learning 
### Modality experts pretraining

Run the following script:

```
python pipeline_expert_pretraining.py
```

Corresponding configuration file:

```
configs_expert_pretraining.py
```

Modify this file as needed (data paths, save paths, UNet settings, learning rate, etc.).  
Note that all four modality experts need to be pretrained one-by-one. However, you may skip this stage and directly go to stage 2 (use higher learning rate though) for time saving. Minor performance degradation can be observed in this case (see Table A2 in the Supplementary for details).


---

## Running Stage 2: Cooperative Learning 
### DMoME training with MoFe ranking loss

Run:

```
python pipeline_joint_training.py
```

Corresponding configuration file:

```
configs_joint_training.py
```

Modify this file as needed (data paths, save paths, modality expert paths, learning rate, etc.).

---

## Citation

If you find our work helpful, please cite our paper:

```bibtex
@inproceedings{li2025simmlm,
  title={Simmlm: A simple framework for multi-modal learning with missing modality},
  author={Li, Sijie and Chen, Chen and Han, Jungong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={24068--24077},
  year={2025}
}
```
