import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceHelper
from monai.inferers import SlidingWindowInferer
import shutil
import os
import numpy as np
import json
import random
from datetime import datetime
import SimpleITK as sitk

from dataset.processors import SingleStreamDataset, BratsEvalSet, LldMmriDataset, LldMmriPreprocessedDataset
from configs_expert_pretraining import DatasetConfig, UNetConfig, TrainingConfig
from train.trainer_expert_pretraining import train_model
from models.nnunet import UNet
from loss.dice_bce_loss import DiceBCEWithLogitsLoss
from pytorch_lightning import seed_everything


def _get_dataset_cls():
    if DatasetConfig.DATASET_NAME == 'lld-mmri':
        if DatasetConfig.USE_PREPROCESSED:
            return LldMmriPreprocessedDataset
        return LldMmriDataset
    return SingleStreamDataset


def _get_dataset_kwargs():
    if DatasetConfig.DATASET_NAME == 'lld-mmri':
        dataset_kwargs = {
            'split_ratios': DatasetConfig.SPLIT_RATIOS,
            'seed': TrainingConfig.RANDOM_SEED,
        }
        if not DatasetConfig.USE_PREPROCESSED:
            dataset_kwargs.update({
                'modalities': DatasetConfig.MODALITIES,
                'label_modality': DatasetConfig.LABEL_MODALITY,
            })
        return dataset_kwargs
    return {}


def get_val_ds():
    dataset_cls = _get_dataset_cls()
    dataset_kwargs = _get_dataset_kwargs()
    dataset_dir = DatasetConfig.PREPROCESSED_DIR if DatasetConfig.USE_PREPROCESSED else DatasetConfig.DATASET_DIR
    if DatasetConfig.VAL_DROP_MODE == 'all':
        val_ds = None
        for dropped_mods in DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS:
            cur_ds = dataset_cls(
                sample_type='val', dataset_dir=dataset_dir,
                splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
                drop_mode=dropped_mods,
                possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
                fold=DatasetConfig.FOLD, unimodality=True,
                **dataset_kwargs,
            )
            if val_ds is None:
                val_ds = cur_ds
            else:
                val_ds = torch.utils.data.ConcatDataset([val_ds, cur_ds])
    else:
        val_ds = dataset_cls(
            sample_type='val', dataset_dir=dataset_dir, splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
            drop_mode=DatasetConfig.VAL_DROP_MODE,
            possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
            fold=DatasetConfig.FOLD, unimodality=True,
            **dataset_kwargs,
        )

    return val_ds


def run_eval():
    if DatasetConfig.DATASET_NAME != 'brats':
        print('Skipping BraTS eval set (not applicable for current dataset).')
        return
    for model_name in ['ckpt_bst.pt', 'ckpt_final.pt']:

        net = UNet(
            input_channels=1,
            n_classes=UNetConfig.N_CLASSES,
            n_stages=UNetConfig.N_STAGES,
            n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
            kernel_size=UNetConfig.KERNEL_SIZES,
            strides=UNetConfig.STRIDES,
            apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION
        ).cuda()
        net.load_state_dict(torch.load(os.path.join(TrainingConfig.RESULTS_DIR, model_name)))
        net = net.eval()

        save_dir = os.path.join(TrainingConfig.RESULTS_DIR, 'bst' if 'bst' in model_name else 'final')

        dropped_mods = DatasetConfig.VAL_DROP_MODE

        ds = BratsEvalSet(dataset_dir=DatasetConfig.EVAL_SET_DIR, drop_mode=dropped_mods, unimodality=True)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        inferer = SlidingWindowInferer(roi_size=[128] * 3, progress=True)

        for i, sample in enumerate((dl)):
            sample_id = sample['sample_id'][0]
            x = sample['image'].cuda()
            crop_coords_0 = sample['crop_coords_0'][0]
            crop_coords_1 = sample['crop_coords_1'][0]

            with torch.no_grad():
                logit = inferer(x, net)
            torch.cuda.empty_cache()
            del x
            pred = torch.sigmoid(logit)[0].detach().cpu().numpy()
            pred = np.where(pred > 0.5, 1, 0)

            pred_WT = pred[0, :, :, :]
            pred_TC = pred[1, :, :, :]
            pred_ET = pred[2, :, :, :]
            seg = np.zeros_like(pred_WT)
            seg = np.where(pred_WT, 2, seg)
            seg = np.where(pred_TC, 1, seg)
            seg = np.where(pred_ET, 4, seg)

            seg = np.pad(
                seg,
                (
                    (crop_coords_0[0],
                     155 - crop_coords_1[0]),
                    (crop_coords_0[1], 240 - crop_coords_1[1]),
                    (crop_coords_0[2], 240 - crop_coords_1[2])
                ),
                mode='constant'
            )

            os.makedirs(os.path.join(save_dir, str(dropped_mods)), exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(seg),
                            fileName=os.path.join(save_dir, str(dropped_mods), sample_id + '.nii.gz'))
    return


def main():
    print('########################################################################')
    print('Current configs:')
    f = open('configs_expert_pretraining.py', 'r')
    print(f.read())
    if not os.path.exists(TrainingConfig.RESULTS_DIR):
        os.makedirs(TrainingConfig.RESULTS_DIR)
    shutil.copyfile(
        'configs_expert_pretraining.py',
        os.path.join(TrainingConfig.RESULTS_DIR,
                     f'current_configs_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.py')
    )
    print('########################################################################')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_everything(seed=TrainingConfig.RANDOM_SEED)

    unet = UNet(
        input_channels=1,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION
    ).to(device=device)

    dataset_cls = _get_dataset_cls()
    dataset_kwargs = _get_dataset_kwargs()
    dataset_dir = DatasetConfig.PREPROCESSED_DIR if DatasetConfig.USE_PREPROCESSED else DatasetConfig.DATASET_DIR
    train_ds = dataset_cls(
        sample_type='train', dataset_dir=dataset_dir, splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=DatasetConfig.FOLD, unimodality=True,
        **dataset_kwargs,
    )

    train_dl = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=4,
    )

    val_ds = get_val_ds()
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    loss_fn = DiceBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=TrainingConfig.LEARNING_RATE)
    # metrics_dict = {"dice": DiceMetric()}
    metrics_dict = {"dice": DiceHelper(include_background=True, sigmoid=True, activate=True, get_not_nans=False,
                                       reduction='mean'), }

    train_model(
        unet,
        optimizer,
        loss_fn,
        metrics_dict,
        val_freq=5,
        train_data=train_dl,
        val_data=val_dl,
        num_epoch=TrainingConfig.N_EPOCHS,
        results_dir=TrainingConfig.RESULTS_DIR,
        apply_early_stopping=TrainingConfig.APPLY_EARLY_STOPPING,
        device=device
    )

    unet = UNet(
        input_channels=1,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION
    ).to(device=device)
    unet.load_state_dict(torch.load(os.path.join(TrainingConfig.RESULTS_DIR, 'ckpt_bst.pt')))
    unet = unet.eval()

    dice = DiceHelper(include_background=True, sigmoid=True, activate=True, get_not_nans=False, reduction='none')

    eval_res = {}

    test_ds = dataset_cls(
        sample_type='test', dataset_dir=dataset_dir, splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.VAL_DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=DatasetConfig.FOLD, unimodality=True,
        **dataset_kwargs,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    inferer = SlidingWindowInferer(roi_size=[128] * 3, progress=True)
    res = []

    for i, sample in enumerate((test_dl)):
        x = sample['img'].to(device)
        y = sample['label'].to(device)
        with torch.no_grad():
            pred = inferer(x, unet)
        torch.cuda.empty_cache()
        del x
        res.append(dice(pred, y).data.cpu().numpy())

    res = np.array(res)
    eval_res[str(DatasetConfig.VAL_DROP_MODE)] = str(np.nanmean(res, axis=0)[0])

    with open(os.path.join(TrainingConfig.RESULTS_DIR, 'eval_res.json'), 'w') as f:
        json.dump(eval_res, f)

    # run_eval()

    return


if __name__ == '__main__':
    main()
