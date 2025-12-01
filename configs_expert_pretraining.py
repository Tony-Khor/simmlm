class UNetConfig:
    INPUT_CHANNELS = 4
    N_CLASSES = 3
    N_STAGES = 6
    # # original nnunet setting
    # N_FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]

    # light-weight one
    N_FEATURES_PER_STAGE = [8, 16, 32, 64, 80, 80]
    KERNEL_SIZES = [[3, 3, 3]] * 6
    STRIDES = [[1, 1, 1], * [[2, 2, 2]] * 5]
    APPLY_DEEP_SUPERVISION = False


class DatasetConfig:
    DATASET_DIR = 'assets/data/nnUNet_preprocessed_BraTS2018'
    SPLITS_FILE_PATH = 'assets/kfold_splits.json'
    EVAL_SET_DIR = 'assets/data/BraTS2018_eval'

    # DROP_MODE: Which modalities are dropped. For modality expert pretraining, you should train the four modality experts using [0, 1, 2] / [0, 1, 3] / [0, 2, 3] / [1, 2, 3]
    DROP_MODE = [0, 1, 3]
    VAL_DROP_MODE = DROP_MODE
    POSSIBLE_DROPPED_MODALITY_COMBINATIONS = [
        [], [0], [1], [2], [3],
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3],
        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
    ]
    # K-fold id; Here we only run the 0th fold
    FOLD = 0


class TrainingConfig:
    RANDOM_SEED = 12345
    N_EPOCHS = 500
    LEARNING_RATE = 0.01
    RESULTS_DIR = 'saved_models/modality_expert_2'
    APPLY_EARLY_STOPPING = False
    APPLY_SAMPLE_WEIGHTS = False
