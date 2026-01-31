import numpy as np
import torch
import json
import os
import SimpleITK as sitk
import random
from copy import deepcopy
from monai.transforms import (ConvertToMultiChannelBasedOnBratsClassesd, Compose, SpatialPadd, RandFlipd, \
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd, RandScaleIntensityd, RandSpatialCropd, \
    CenterSpatialCropd, CropForeground, CropForegroundd)
from torch.utils.data import Dataset
from typing import List, Union
from scipy.ndimage import binary_fill_holes

from dataset.utils import zero_mean_unit_variance_normalization
from dataset.utils import z_score_norm_with_mask


def _load_split_ids(sample_ids, splits_file_path, sample_type, fold, seed=12345, split_ratios=(0.8, 0.1, 0.1)):
    if splits_file_path and os.path.exists(splits_file_path):
        with open(splits_file_path, 'r') as f:
            split_data = json.load(f)
        if isinstance(split_data, list):
            split_data = split_data[fold]
        if sample_type in split_data:
            return split_data[sample_type]
        raise KeyError(f'No "{sample_type}" key found in {splits_file_path}')

    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f'split_ratios must sum to 1.0, got {split_ratios}')

    rng = np.random.default_rng(seed)
    sample_ids = sorted(sample_ids)
    rng.shuffle(sample_ids)
    n_total = len(sample_ids)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    split_map = {
        'train': sample_ids[:n_train],
        'val': sample_ids[n_train:n_train + n_val],
        'test': sample_ids[n_train + n_val:],
    }
    return split_map[sample_type]


def _build_lld_mmri_cases(images_dir, labels_dir, modalities, label_modality):
    suffix_lookup = {modality: f'_{modality}_0000.nii.gz' for modality in modalities}
    case_map = {}

    for fname in os.listdir(images_dir):
        for modality, suffix in suffix_lookup.items():
            if fname.endswith(suffix):
                case_id = fname[:-len(suffix)]
                case_map.setdefault(case_id, {})[modality] = os.path.join(images_dir, fname)
                break

    valid_cases = {}
    for case_id, modality_map in case_map.items():
        if len(modality_map) != len(modalities):
            continue
        label_path = os.path.join(labels_dir, f'{case_id}_{label_modality}.nii.gz')
        if not os.path.exists(label_path):
            continue
        valid_cases[case_id] = {
            'images': [modality_map[modality] for modality in modalities],
            'label': label_path,
        }

    return valid_cases


class LldMmriDataset(Dataset):
    def __init__(
            self,
            sample_type: str,
            dataset_dir: str,
            splits_file_path: str,
            drop_mode: Union[None, str, List],
            possible_dropped_modality_combinations: List,
            modalities: List,
            label_modality: str,
            fold: int = 0,
            unimodality: bool = False,
            split_ratios=(0.8, 0.1, 0.1),
            seed: int = 12345,
    ):
        assert sample_type in [
            'train', 'val', 'test'
        ], f'Invalid sample type: {sample_type}. Must be one of ["train", "val", "test"]'

        if unimodality:
            expected_drop = len(modalities) - 1
            assert isinstance(drop_mode, List) and len(drop_mode) == expected_drop, \
                f'If unimodality==True, drop_mode must be a size-{expected_drop} List'

        self.sample_type = sample_type
        self.dataset_dir = dataset_dir
        self.drop_mode = drop_mode
        self.possible_dropped_modality_combinations = possible_dropped_modality_combinations
        self.modalities = modalities
        self.label_modality = label_modality
        self.unimodality = unimodality
        self.seed = seed

        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            raise FileNotFoundError(f'Expected images/labels dirs under {dataset_dir}')

        self.case_map = _build_lld_mmri_cases(images_dir, labels_dir, modalities, label_modality)
        if len(self.case_map) == 0:
            raise RuntimeError('No valid cases found for LLD-MMRI dataset.')

        self.sample_ls = _load_split_ids(
            list(self.case_map.keys()),
            splits_file_path=splits_file_path,
            sample_type=sample_type,
            fold=fold,
            seed=seed,
            split_ratios=split_ratios,
        )

        self.sample_transforms = self._get_sample_transforms()

    def __len__(self):
        return len(self.sample_ls)

    def _get_sample_transforms(self):
        if self.sample_type == 'train':
            sample_transforms = [
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
                RandSpatialCropd(keys=['img', 'label'], roi_size=[128] * 3),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=0),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=1),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=2),
                RandGaussianNoised(keys='img', prob=.15, mean=.0, std=.33 * random.random()),
                RandGaussianSmoothd(keys='img', prob=.15, sigma_x=(.5, 1.5), sigma_y=(.5, 1.5), sigma_z=(.5, 1.5)),
                RandAdjustContrastd(keys='img', prob=.15, gamma=(.7, 1.4)),
                RandScaleIntensityd(keys='img', prob=.15, factors=(0.7, 1.4))
            ]
        elif self.sample_type == 'val':
            sample_transforms = [
                CenterSpatialCropd(keys=['img', 'label'], roi_size=[128] * 3),
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]
        else:
            sample_transforms = [
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]

        return Compose(sample_transforms)

    def _load_case(self, case_id):
        case_info = self.case_map[case_id]
        images = [
            sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
            for path in case_info['images']
        ]
        img = np.stack([zero_mean_unit_variance_normalization(mod) for mod in images])

        label = sitk.GetArrayFromImage(sitk.ReadImage(case_info['label'])).astype(np.float32)
        label = (label > 0).astype(np.float32)
        label = label[None, ...]
        return img, label

    def __getitem__(self, idx):
        case_id = self.sample_ls[idx]
        img, label = self._load_case(case_id)

        sample = {
            'img': img,
            'label': label,
        }

        sample = self.sample_transforms(sample)

        num_modalities = img.shape[0]
        sample['mask_code'] = torch.ones(num_modalities)

        if isinstance(self.drop_mode, str) and self.drop_mode == 'rand':
            drop_mods = random.choice(self.possible_dropped_modality_combinations)
            sample['mask_code'] = torch.tensor([0 if _ in drop_mods else 1 for _ in range(num_modalities)])
            sample['img'][drop_mods, ...] = 0
        elif isinstance(self.drop_mode, List):
            sample['mask_code'] = torch.tensor([0 if _ in self.drop_mode else 1 for _ in range(num_modalities)])
            if self.unimodality:
                for c in range(num_modalities):
                    if c not in self.drop_mode:
                        sample['img'] = sample['img'][c: c + 1]
            else:
                sample['img'][self.drop_mode, ...] = 0
        elif self.drop_mode is not None:
            raise NotImplementedError

        sample['mask_encoding'] = (
            torch.sum(sample['mask_code'] * torch.tensor([2 ** idx for idx in range(num_modalities)]))
        ).to(torch.int64)
        sample['weight'] = 2 if torch.sum(sample['mask_code']) == 1 else 1
        sample['sample_id'] = case_id

        return sample


class LldMmriPairedDataset(LldMmriDataset):
    def __getitem__(self, idx):
        case_id = self.sample_ls[idx]
        img, label = self._load_case(case_id)

        sample = {
            'img': img,
            'label': label,
        }

        sample_ = deepcopy(sample)

        sample = self.sample_transforms(sample)
        sample_ = self.sample_transforms(sample_)

        num_modalities = img.shape[0]
        assert self.drop_mode == 'rand', 'Only support "rand" mode for training'

        dm1 = random.choice(self.possible_dropped_modality_combinations)
        dm2 = random.choice(self.possible_dropped_modality_combinations)
        while not (set(dm1) < set(dm2) or set(dm2) < set(dm1)) or dm1 == []:
            dm1 = random.choice(self.possible_dropped_modality_combinations)
            dm2 = random.choice(self.possible_dropped_modality_combinations)
        if set(dm1) > set(dm2):
            dm1, dm2 = dm2, dm1

        sample['mask_code'] = torch.tensor([0 if _ in dm1 else 1 for _ in range(num_modalities)])
        sample['img'][dm1] = 0
        sample_['mask_code'] = torch.tensor([0 if _ in dm2 else 1 for _ in range(num_modalities)])
        sample_['img'][dm2] = 0

        sample['img_'] = sample_['img']
        sample['label_'] = sample_['label']
        sample['mask_code_'] = sample_['mask_code']

        sample['mask_encoding'] = (
            torch.sum(sample['mask_code'] * torch.tensor([2 ** idx for idx in range(num_modalities)]))
        ).to(torch.int64)
        sample['weight'] = 2 if torch.sum(sample['mask_code']) == 1 else 1
        sample['sample_id'] = case_id

        return sample


# Regular BraTS dataset with randomly droppe modalities
class SingleStreamDataset(Dataset):
    def __init__(
            self, sample_type: str, dataset_dir: str, splits_file_path: str,
            drop_mode: Union[None, str, List], possible_dropped_modality_combinations: List,
            fold: int = 0, unimodality=False
    ):
        assert sample_type in [
            'train', 'val', 'test'
        ], f'Invalid sample type: {sample_type}. Must be one of ["train", "val", "test"]'

        assert (not unimodality) or (unimodality and isinstance(drop_mode, List) and len(drop_mode) == 3) \
            , f'If unimodality==True, drop_mode must be a size-3 List'

        self.sample_type = sample_type
        self.dataset_dir = dataset_dir
        self.drop_mode = drop_mode
        self.possible_dropped_modality_combinations = possible_dropped_modality_combinations
        self.unimodality = unimodality

        with open(splits_file_path, 'r') as f:
            self.sample_ls = json.load(f)[fold]['train' if sample_type == 'train' else 'val']

        self.sample_transforms = self._get_sample_transforms()

    def __len__(self):

        return len(self.sample_ls)

    def _get_sample_transforms(self):
        """
        Get set of monai transforms according to the sample type
        """
        if self.sample_type == 'train':
            sample_transforms = [
                # padding the image in case that size of any dimension is smaller than1 128 after foreground cropping
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
                # crop the sample to 128*128*128
                # BiasedCropper(keys=['img', 'label'], label_key='label', spatial_size=[128] * 3, pos=1, neg=1,
                #               image_key='img'),
                RandSpatialCropd(keys=['img', 'label'], roi_size=[128]*3),

                # spatial augmentations
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=0),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=1),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=2),

                # intensity augmentations
                RandGaussianNoised(keys='img', prob=.15, mean=.0, std=.33 * random.random()),
                RandGaussianSmoothd(keys='img', prob=.15, sigma_x=(.5, 1.5), sigma_y=(.5, 1.5), sigma_z=(.5, 1.5)),
                RandAdjustContrastd(keys='img', prob=.15, gamma=(.7, 1.4)),
                RandScaleIntensityd(keys='img', prob=.15, factors=(0.7, 1.4))
            ]
        elif self.sample_type == 'val':
            sample_transforms = [
                CenterSpatialCropd(keys=['img', 'label'], roi_size=[128]*3),
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]
        else:
            sample_transforms = [
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]

        return sample_transforms

    def _transform_label(self, seg):
        wt_seg = torch.logical_or(torch.logical_or(seg == 1, seg == 2), seg == 3)
        tc_seg = torch.logical_or(seg == 2, seg == 3)
        et_seg = seg == 3

        mask = seg == 0

        return torch.cat((wt_seg, tc_seg, et_seg)), torch.cat((mask, mask, mask, mask))

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.dataset_dir, self.sample_ls[idx] + '.npy'))
        label = np.load(os.path.join(self.dataset_dir, self.sample_ls[idx] + '_seg.npy'))

        label[label==0] = -2
        label[label==-1] = 0

        sample = {
            'img': img,
            'label': label,
        }

        sample = Compose(self.sample_transforms)(sample)

        label, mask = self._transform_label(sample['label'])
        sample['img'][mask] = 0
        sample['background_mask'] = mask
        sample['label'] = label
        sample['mask_code'] = torch.ones(img.shape[0])

        if isinstance(self.drop_mode, str) and self.drop_mode == 'rand':
            drop_mods = random.choice(self.possible_dropped_modality_combinations)
            sample['mask_code'] = torch.tensor([0 if _ in drop_mods else 1 for _ in range(img.shape[0])])
            sample['img'][drop_mods, ...] = 0
        elif isinstance(self.drop_mode, List):
            sample['mask_code'] = torch.tensor([0 if _ in self.drop_mode else 1 for _ in range(img.shape[0])])
            if self.unimodality:
                for c in range(4):
                    if c not in self.drop_mode:
                        sample['img'] = sample['img'][c: c+1]
            else:
                sample['img'][self.drop_mode, ...] = 0
        elif self.drop_mode is not None:
            raise NotImplementedError

        sample['mask_encoding'] = (torch.sum(sample['mask_code'] * torch.tensor([1, 2, 4, 8]))).to(torch.int64)
        sample['weight'] = 2 if torch.sum(sample['mask_code']) == 1 else 1
        sample['sample_id'] = self.sample_ls[idx]

        return sample


# Dataset for BraTS eval set (for the official evaluation)
class BratsEvalSet(Dataset):
    def __init__(self, dataset_dir: str, drop_mode: List, unimodality=False):
        assert (not unimodality) or (unimodality and isinstance(drop_mode, List) and len(drop_mode) == 3) \
            , f'If unimodality==True, drop_mode must be a size-3 List'

        self.dataset_dir = dataset_dir
        self.drop_mode = drop_mode
        self.sample_dir_list = os.listdir(dataset_dir)
        self.unimodality = unimodality

    def __len__(self):
        return len(self.sample_dir_list)

    def __getitem__(self, idx):
        sample_dir = self.sample_dir_list[idx]
        file_list = os.listdir(os.path.join(self.dataset_dir, sample_dir))
        t1 = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 't1.nii' in _))
        t1ce = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 't1ce.nii' in _))
        t2 = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 't2.nii' in _))
        flair = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 'flair.nii' in _))

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1)).astype(np.float32)
        t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce)).astype(np.float32)
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2)).astype(np.float32)
        flair = sitk.GetArrayFromImage(sitk.ReadImage(flair)).astype(np.float32)

        img = np.stack([t1, t1ce, t2, flair])

        img[self.drop_mode] = 0
        img, crop_coords_0, crop_coords_1 = CropForeground(return_coords=True)(img)

        mask = np.max(img, axis=0) != 0
        mask = binary_fill_holes(mask)
        img = np.stack([z_score_norm_with_mask(img[_], mask) for _ in range(img.shape[0])])

        if self.unimodality:
            for c in range(4):
                if c not in self.drop_mode:
                    img = img[c: c+1],
        else:
            img[self.drop_mode] = 0

        d_info = {
            'sample_id': sample_dir,
            'image': torch.from_numpy(img),
            'crop_coords_0': crop_coords_0,
            'crop_coords_1': crop_coords_1,
            'mask_code': torch.tensor([0 if _ in self.drop_mode else 1 for _ in range(img.shape[0])])
        }

        return d_info


# Paired BraTS dataset which has samples with more modalities and fewer modalities; it's for MoFe loss computation
class PairedDataset(Dataset):
    def __init__(
            self, sample_type: str, dataset_dir: str, splits_file_path: str,
            drop_mode: Union[None, str, List], possible_dropped_modality_combinations: List, fold: int = 0,
    ):
        assert sample_type in [
            'train', 'val', 'test'
        ], f'Invalid sample type: {sample_type}. Must be one of ["train", "val", "test"]'

        self.sample_type = sample_type
        self.dataset_dir = dataset_dir
        self.drop_mode = drop_mode
        self.possible_dropped_modality_combinations = possible_dropped_modality_combinations

        with open(splits_file_path, 'r') as f:
            self.sample_ls = json.load(f)[fold]['train' if sample_type == 'train' else 'val']

        self.sample_transforms = self._get_sample_transforms()

    def __len__(self):

        return len(self.sample_ls)

    def _get_sample_transforms(self):
        """
        Get set of monai transforms according to the sample type
        """
        if self.sample_type == 'train':
            sample_transforms = [
                # padding the image in case that size of any dimension is smaller than1 128 after foreground cropping
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
                # crop the sample to 128*128*128
                # BiasedCropper(keys=['img', 'label'], label_key='label', spatial_size=[128] * 3, pos=1, neg=1,
                #               image_key='img'),
                RandSpatialCropd(keys=['img', 'label'], roi_size=[128] * 3),

                # spatial augmentations
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=0),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=1),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=2),

                # intensity augmentations
                RandGaussianNoised(keys='img', prob=.15, mean=.0, std=.33 * random.random()),
                RandGaussianSmoothd(keys='img', prob=.15, sigma_x=(.5, 1.5), sigma_y=(.5, 1.5), sigma_z=(.5, 1.5)),
                RandAdjustContrastd(keys='img', prob=.15, gamma=(.7, 1.4)),
                RandScaleIntensityd(keys='img', prob=.15, factors=(0.7, 1.4))
            ]
        elif self.sample_type == 'val':
            sample_transforms = [
                CenterSpatialCropd(keys=['img', 'label'], roi_size=[128] * 3),
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]
        else:
            sample_transforms = [
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]

        return Compose(sample_transforms)

    def _transform_label(self, seg):
        wt_seg = torch.logical_or(torch.logical_or(seg == 1, seg == 2), seg == 3)
        tc_seg = torch.logical_or(seg == 2, seg == 3)
        et_seg = seg == 3

        mask = seg == 0

        return torch.cat((wt_seg, tc_seg, et_seg)), torch.cat((mask, mask, mask, mask))

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.dataset_dir, self.sample_ls[idx] + '.npy'))
        label = np.load(os.path.join(self.dataset_dir, self.sample_ls[idx] + '_seg.npy'))

        label[label == 0] = -2
        label[label == -1] = 0

        sample = {
            'img': img,
            'label': label,
        }

        if self.sample_type == 'train':
            sample_ = deepcopy(sample)
            sample_ = self.sample_transforms(sample_)

        sample = self.sample_transforms(sample)

        label, mask = self._transform_label(sample['label'])
        sample['img'][mask] = 0
        sample['label'] = label

        if self.sample_type == 'train':
            sample['img_'] = sample_['img']
            label_, mask = self._transform_label(sample_['label'])
            sample['img_'][mask] = 0
            sample['label_'] = label_

            assert self.drop_mode == 'rand', 'Only support "rand" mode for training'
            # Randomly select modality-drop combinations to form more-modality samples and fewer-modality samples required by MoFe
            dm1 = random.choice(self.possible_dropped_modality_combinations)
            dm2 = random.choice(self.possible_dropped_modality_combinations)
            '''
            Some explanation for the potential confusions.
            
            This WHILE loop is designed to ensure that dm1 is a subset of dm2 or dm2 is a subset of dm1 (to satisfy MoFe), 
            while also ensuring that dm1 is not the “select-all” modality (i.e., dm1 ≠ []) 
            (to decrease the select-all modality combination in this specific sampling strategy, 
            so that the number of randomly sampled modality combinations remains mostly balanced/uniform).
            
            If you run simulations with repeated random sampling with these code, 
            you’ll find that this sampling strategy results in roughly equal selection rates across all modality combinations.

            It’s worth mentioning that, empirically, this selection method does not harm model performance. 
            We encourage the community to explore more optimized sampling strategies (including MoFe's) for missing modality problem.
            '''
            while not (set(dm1) < set(dm2) or set(dm2) < set(dm1)) or dm1 == []:
                dm1 = random.choice(self.possible_dropped_modality_combinations)
                dm2 = random.choice(self.possible_dropped_modality_combinations)
            if set(dm1) > set(dm2):
                dm1, dm2 = dm2, dm1

            sample['mask_code'] = torch.tensor([0 if _ in dm1 else 1 for _ in range(img.shape[0])])
            sample['img'][dm1] = 0
            sample['mask_code_'] = torch.tensor([0 if _ in dm2 else 1 for _ in range(img.shape[0])])
            sample['img_'][dm2] = 0
        else:
            if isinstance(self.drop_mode, str) and self.drop_mode == 'rand':
                drop_mods = random.choice(self.possible_dropped_modality_combinations)
                sample['mask_code'] = torch.tensor([0 if _ in drop_mods else 1 for _ in range(img.shape[0])])
                sample['img'][drop_mods] = 0
            elif isinstance(self.drop_mode, List):
                sample['mask_code'] = torch.tensor([0 if _ in self.drop_mode else 1 for _ in range(img.shape[0])])
                sample['img'][self.drop_mode] = 0
            elif self.drop_mode is not None:
                raise NotImplementedError

        return sample
