class ModelConfig:
    # DMoMEOutputLevel, DMoMEProbLevel, DMoMEFeatureLevel, MoMKE
    MODEL = 'DMoMEOutputLevel'
    INPUT_CHANNELS = 4
    N_CLASSES = 3
    N_STAGES = 6
    # # original nnunet setting
    # N_FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]

    # light-weight one
    N_FEATURES_PER_STAGE = [8, 16, 32, 64, 80, 80]
    KERNEL_SIZES = [[3, 3, 3]] * 6
    STRIDES = [[1, 1, 1], *[[2, 2, 2]] * 5]
    # Use pre-trained modality experts to initialize the DMoME. You may also try skip the expert pretraining stage by setting the list as [None] * 4
    PRETRAINED_EXPERT_FILE_LIST = [
        'saved_models/modality_expert_0/ckpt_bst.pt',
        'saved_models/modality_expert_1/ckpt_bst.pt',
        'saved_models/modality_expert_2/ckpt_bst.pt',
        'saved_models/modality_expert_3/ckpt_bst.pt',
    ]

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
    DATASET_DIR = 'assets/data/nnUNet_preprocessed_BraTS2018'
    SPLITS_FILE_PATH = 'assets/kfold_splits.json'
    EVAL_SET_DIR = 'assets/data/BraTS2018_eval'

    DROP_MODE = 'rand'
    VAL_DROP_MODE = 'all'
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
    N_EPOCHS = 300
    # LEARNING_RATE = 0.01
    LEARNING_RATE = 0.001
    RESULTS_DIR = 'saved_models/dmome'
    APPLY_EARLY_STOPPING = False
    APPLY_SAMPLE_WEIGHTS = False
