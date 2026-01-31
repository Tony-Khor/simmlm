from itertools import combinations


class UNetConfig:
    INPUT_CHANNELS = 1
    N_CLASSES = 1
    N_STAGES = 6
    # # original nnunet setting
    # N_FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]

    # light-weight one
    N_FEATURES_PER_STAGE = [8, 16, 32, 64, 80, 80]
    KERNEL_SIZES = [[3, 3, 3]] * 6
    STRIDES = [[1, 1, 1], * [[2, 2, 2]] * 5]
    APPLY_DEEP_SUPERVISION = False


class DatasetConfig:
    DATASET_NAME = 'lld-mmri'
    DATASET_DIR = 'assets/data/lld-mmri'
    PREPROCESSED_DIR = 'assets/data/lld-mmri_preprocessed'
    USE_PREPROCESSED = False
    SPLITS_FILE_PATH = None
    EVAL_SET_DIR = None
    MODALITIES = ['C+A', 'C+Delay', 'C+V', 'C-pre', 'DWI', 'InPhase', 'OutPhase', 'T2WI']
    LABEL_MODALITY = 'C+A'
    SPLIT_RATIOS = (0.8, 0.1, 0.1)

    # DROP_MODE: Which modalities are dropped. For modality expert pretraining, set seven indices to keep one modality.
    DROP_MODE = [0, 1, 2, 3, 4, 5, 6]
    VAL_DROP_MODE = DROP_MODE
    NUM_MODALITIES = len(MODALITIES)
    POSSIBLE_DROPPED_MODALITY_COMBINATIONS = [
        list(combo)
        for r in range(0, NUM_MODALITIES + 1)
        for combo in combinations(range(NUM_MODALITIES), r)
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
