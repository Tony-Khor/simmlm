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
from pytorch_lightning.utilities.seed import seed_everything

from dataset.processors import SingleStreamDataset, BratsEvalSet, PairedDataset, LldMmriDataset, LldMmriPairedDataset
from configs_joint_training import DatasetConfig, TrainingConfig, ModelConfig
from train.trainer_joint_training import train_model
from models.dmome import DMoMEOutputLevel, DMoMEFeatureLevel, DMoMEProbLevel
from models.momke import MoMKE


def _get_dataset_cls():
    if DatasetConfig.DATASET_NAME == 'lld-mmri':
        return LldMmriDataset
    return SingleStreamDataset


def _get_train_dataset_cls():
    if DatasetConfig.DATASET_NAME == 'lld-mmri':
        return LldMmriPairedDataset
    return PairedDataset


def _get_dataset_kwargs():
    if DatasetConfig.DATASET_NAME == 'lld-mmri':
        return {
            'modalities': DatasetConfig.MODALITIES,
            'label_modality': DatasetConfig.LABEL_MODALITY,
            'split_ratios': DatasetConfig.SPLIT_RATIOS,
            'seed': TrainingConfig.RANDOM_SEED,
        }
    return {}


def get_val_ds():
    dataset_cls = _get_dataset_cls()
    dataset_kwargs = _get_dataset_kwargs()
    if DatasetConfig.VAL_DROP_MODE == 'all':
        val_ds = None
        for dropped_mods in DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS:
            cur_ds = dataset_cls(
                sample_type='val', dataset_dir=DatasetConfig.DATASET_DIR,
                splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
                drop_mode=dropped_mods,
                possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
                fold=DatasetConfig.FOLD,
                **dataset_kwargs,
            )
            if val_ds is None:
                val_ds = cur_ds
            else:
                val_ds = torch.utils.data.ConcatDataset([val_ds, cur_ds])
    else:
        val_ds = dataset_cls(
            sample_type='val', dataset_dir=DatasetConfig.DATASET_DIR, splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
            drop_mode=DatasetConfig.VAL_DROP_MODE,
            possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
            fold=DatasetConfig.FOLD,
            **dataset_kwargs,
        )

    return val_ds


def load_model():
    assert ModelConfig.MODEL in [
        'MoMKE', 'DMoMEOutputLevel', 'DMoMEProbLevel', 'DMoMEFeatureLevel',
    ], f'Invalid model type: {ModelConfig.MODEL}.'
    if ModelConfig.MODEL == 'MoMKE':
        return MoMKE().cuda()
    elif ModelConfig.MODEL == 'DMoMEOutputLevel':
        return DMoMEOutputLevel().cuda()
    elif ModelConfig.MODEL == 'DMoMEProbLevel':
        return DMoMEProbLevel().cuda()
    elif ModelConfig.MODEL == 'DMoMEFeatureLevel':
        return DMoMEFeatureLevel().cuda()


# Run and save BraTS evaluation set. The saved segmentation results can be uploaded to BraTS official evaluation platform for metrics.
def run_eval():
    if DatasetConfig.DATASET_NAME != 'brats':
        print('Skipping BraTS eval set (not applicable for current dataset).')
        return
    for model_name in ['ckpt_bst.pt', 'ckpt_final.pt']:

        net = load_model()
        net.load_state_dict(torch.load(os.path.join(TrainingConfig.RESULTS_DIR, model_name)))
        net = net.eval()

        save_dir = os.path.join(TrainingConfig.RESULTS_DIR, 'bst' if 'bst' in model_name else 'final')

        for dropped_mods in DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS:
            print(dropped_mods)

            ds = BratsEvalSet(dataset_dir=DatasetConfig.EVAL_SET_DIR, drop_mode=dropped_mods)
            dl = DataLoader(ds, batch_size=1, shuffle=False)

            inferer = SlidingWindowInferer(roi_size=[128] * 3, progress=True)

            for i, sample in enumerate((dl)):
                sample_id = sample['sample_id'][0]
                x = sample['image'].cuda()
                crop_coords_0 = sample['crop_coords_0'][0]
                crop_coords_1 = sample['crop_coords_1'][0]

                with torch.no_grad():
                    o = inferer(x, net)
                torch.cuda.empty_cache()
                del x

                if ModelConfig.VAL_LOSS_ARGS['need_sigmoid']:
                    pred = torch.sigmoid(o)[0].detach().cpu().numpy()
                else:
                    pred = o[0].detach().cpu().numpy()
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
    torch.autograd.set_detect_anomaly(True)
    seed_everything(TrainingConfig.RANDOM_SEED)

    print('########################################################################')
    print('Current configs:')
    f = open('configs_joint_training.py', 'r')
    print(f.read())
    if not os.path.exists(TrainingConfig.RESULTS_DIR):
        os.makedirs(TrainingConfig.RESULTS_DIR)
    shutil.copyfile(
        'configs_joint_training.py',
        os.path.join(TrainingConfig.RESULTS_DIR,
                     f'current_configs_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.py')
    )
    print('########################################################################')

    net = load_model()

    dataset_cls = _get_train_dataset_cls()
    dataset_kwargs = _get_dataset_kwargs()
    train_ds = dataset_cls(
        sample_type='train', dataset_dir=DatasetConfig.DATASET_DIR, splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=DatasetConfig.FOLD,
        **dataset_kwargs,
    )

    train_dl = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=4,
    )

    val_ds = get_val_ds()
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(net.parameters(), lr=TrainingConfig.LEARNING_RATE)
    # optimizer = torch.optim.Adam([
    #     {'params': net.router.parameters(), 'lr': TrainingConfig.LEARNING_RATE},
    #     {'params': net.expert_ls.parameters(), 'lr': 0.1 * TrainingConfig.LEARNING_RATE}
    # ])

    history = train_model(
        net,
        optimizer,
        train_data=train_dl,
        val_data=val_dl,
        val_freq=5,
        num_epoch=TrainingConfig.N_EPOCHS,
        results_dir=TrainingConfig.RESULTS_DIR,
        apply_early_stopping=TrainingConfig.APPLY_EARLY_STOPPING,
    )

    for m in ['bst', 'final']:
        net = load_model()
        net.load_state_dict(torch.load(os.path.join(TrainingConfig.RESULTS_DIR, f'ckpt_{m}.pt')))
        unet = net.eval()

        dice = DiceHelper(
            include_background=True, sigmoid=True, softmax=False,
            activate=ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'], get_not_nans=False, reduction='none'
        )

        eval_res = {}

        for dropped_mods in DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS:

            test_ds = dataset_cls(
                sample_type='test', dataset_dir=DatasetConfig.DATASET_DIR,
                splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
                drop_mode=dropped_mods,
                possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
                fold=DatasetConfig.FOLD,
                **dataset_kwargs,
            )
            test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

            inferer = SlidingWindowInferer(roi_size=[128] * 3, progress=True)
            res = []

            for i, sample in enumerate((test_dl)):
                x = sample['img'].cuda()
                y = sample['label'].cuda()
                with torch.no_grad():
                    pred = inferer(x, unet)
                torch.cuda.empty_cache()
                del x
                res.append(dice(pred, y).data.cpu().numpy())

            res = np.array(res)
            eval_res[str(dropped_mods)] = str(np.nanmean(res, axis=0)[0])

        with open(os.path.join(TrainingConfig.RESULTS_DIR, f'eval_res_{m}.json'), 'w') as f:
            json.dump(eval_res, f)

    # run_eval()

    return


if __name__ == '__main__':
    main()
