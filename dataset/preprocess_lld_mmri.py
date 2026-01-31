import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from dataset.utils import zero_mean_unit_variance_normalization


def _discover_cases(images_dir, labels_dir, modalities, label_modality):
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


def _save_case(case_id, case_info, output_dir):
    images = [
        sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        for path in case_info['images']
    ]
    img = np.stack([zero_mean_unit_variance_normalization(mod) for mod in images])

    label = sitk.GetArrayFromImage(sitk.ReadImage(case_info['label'])).astype(np.float32)
    label = (label > 0).astype(np.float32)
    label = label[None, ...]

    np.save(os.path.join(output_dir, f'{case_id}.npy'), img)
    np.save(os.path.join(output_dir, f'{case_id}_seg.npy'), label)


def _write_splits(output_dir, case_ids, split_ratios, seed):
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f'split_ratios must sum to 1.0, got {split_ratios}')

    rng = np.random.default_rng(seed)
    case_ids = sorted(case_ids)
    rng.shuffle(case_ids)
    n_total = len(case_ids)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    split_map = {
        'train': case_ids[:n_train],
        'val': case_ids[n_train:n_train + n_val],
        'test': case_ids[n_train + n_val:],
    }
    with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
        json.dump(split_map, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Preprocess LLD-MMRI dataset into standardized numpy arrays.')
    parser.add_argument('--input-dir', required=True, help='Root folder containing images/ and labels/ subfolders.')
    parser.add_argument('--output-dir', required=True, help='Folder to write standardized .npy files.')
    parser.add_argument('--modalities', nargs='+', required=True, help='Ordered list of modality suffixes.')
    parser.add_argument('--label-modality', required=True, help='Modality suffix used for labels (e.g., C+A).')
    parser.add_argument('--split-ratios', type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--dry-run', action='store_true', help='Validate the dataset structure without writing outputs.')
    args = parser.parse_args()

    images_dir = os.path.join(args.input_dir, 'images')
    labels_dir = os.path.join(args.input_dir, 'labels')
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise FileNotFoundError('Expected images/ and labels/ directories under input-dir.')

    cases = _discover_cases(images_dir, labels_dir, args.modalities, args.label_modality)
    if len(cases) == 0:
        raise RuntimeError('No valid cases discovered. Check modality names and label modality.')

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for case_id, case_info in cases.items():
        if args.dry_run:
            continue
        _save_case(case_id, case_info, str(output_dir))

    if not args.dry_run:
        _write_splits(str(output_dir), list(cases.keys()), args.split_ratios, args.seed)

    print(f'Processed {len(cases)} cases.')


if __name__ == '__main__':
    main()
