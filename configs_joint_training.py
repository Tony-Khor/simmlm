from itertools import combinations


class ModelConfig:
    # DMoMEOutputLevel, DMoMEProbLevel, DMoMEFeatureLevel, MoMKE
    MODEL = 'DMoMEOutputLevel'
    INPUT_CHANNELS = 8
    N_CLASSES = 1
    N_STAGES = 6
    # # original nnunet setting
    # N_FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]

    # light-weight one
    N_FEATURES_PER_STAGE = [8, 16, 32, 64, 80, 80]
    KERNEL_SIZES = [[3, 3, 3]] * 6
    STRIDES = [[1, 1, 1], *[[2, 2, 2]] * 5]
    # Use pre-trained modality experts to initialize the DMoME. You may also try skip the expert pretraining stage by setting the list as [None] * INPUT_CHANNELS
    PRETRAINED_EXPERT_FILE_LIST = [None] * INPUT_CHANNELS

    TRAIN_LOSS_ARGS = {
        'need_sigmoid': True,
        'compute_ranking_loss': True,
        'seg_loss_weight': 1,
        # This is the alhpa for our MoFe loss
        'ranking_loss_weight': 0.1
    }
    VAL_LOSS_ARGS = {
        'need_sigmoid': True,
        'compute_ranking_loss': False,
    }


class DatasetConfig:
    DATASET_NAME = 'lld-mmri'
    DATASET_DIR = 'assets/data/lld-mmri'
    SPLITS_FILE_PATH = None
    EVAL_SET_DIR = None
    MODALITIES = ['C+A', 'C+Delay', 'C+V', 'C-pre', 'DWI', 'InPhase', 'OutPhase', 'T2WI']
    LABEL_MODALITY = 'C+A'
    SPLIT_RATIOS = (0.8, 0.1, 0.1)

    DROP_MODE = 'rand'
    VAL_DROP_MODE = 'all'
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
    N_EPOCHS = 300
    # LEARNING_RATE = 0.01
    LEARNING_RATE = 0.001
    RESULTS_DIR = 'saved_models/dmome'
    APPLY_EARLY_STOPPING = False
    APPLY_SAMPLE_WEIGHTS = False
