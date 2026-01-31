import torch
from torch import nn

from configs_joint_training import ModelConfig
from models.nnunet import UNet


# Implementation for our default DMoME, where the fusion of the experts is operated at the output level (use segmentation map before sigmoid)
class DMoMEOutputLevel(nn.Module):
    def __init__(self, ignore_assert=False):
        super().__init__()
        if ignore_assert:
            print("CAREFUL!!!!!!YOU ARE IGNORING THE ASSERTATION OF MODEL CREATION!!!!!!")
        else:
            assert ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'] and ModelConfig.VAL_LOSS_ARGS['need_sigmoid'], \
                "loss for this model needs sigmoid!"

        self.pretrained_expert_file_list = ModelConfig.PRETRAINED_EXPERT_FILE_LIST
        self.n_stages = ModelConfig.N_STAGES
        self.num_modalities = ModelConfig.INPUT_CHANNELS
        self.num_classes = ModelConfig.N_CLASSES
        if len(self.pretrained_expert_file_list) != self.num_modalities:
            raise ValueError('PRETRAINED_EXPERT_FILE_LIST must match INPUT_CHANNELS')

        self.expert_ls = nn.ModuleList([
            self._build_single_expert(expert_id)
            for expert_id in range(self.num_modalities)
        ])
        self.router = nn.Sequential(
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=self.num_modalities * 64, out_features=self.num_modalities * self.num_classes),
        ).cuda()

        # self.router = nn.Sequential(
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(in_features=256, out_features=64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=64, out_features=4 * 3),
        # ).cuda()

        # self.router = nn.Sequential(
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(in_features=256, out_features=64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=64, out_features=4 * 3),
        # ).cuda()

        self.expert_output_shape = self.expert_ls[0](torch.rand(1, 1, 128, 128, 128).cuda()).shape

    def _build_single_expert(self, expert_id):
        net = UNet(
            input_channels=1,
            n_classes=ModelConfig.N_CLASSES,
            n_stages=ModelConfig.N_STAGES,
            n_features_per_stage=ModelConfig.N_FEATURES_PER_STAGE,
            kernel_size=ModelConfig.KERNEL_SIZES,
            strides=ModelConfig.STRIDES
        ).cuda()

        if self.pretrained_expert_file_list[expert_id] is not None:
            net.load_state_dict(torch.load(self.pretrained_expert_file_list[expert_id]))

        return net

    def forward(self, x):
        modality_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1) # True if modality missing

        weight = self.router(x).view(-1, self.num_modalities, self.num_classes, 1, 1, 1)
        weight[modality_mask] = float('-inf')
        weight = nn.functional.softmax(weight, dim=1)

        output = torch.stack([
            torch.cat([
                torch.zeros(self.expert_output_shape).cuda()
                if modality_mask[sample_idx, modality_idx]
                else self.expert_ls[modality_idx](x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...])
                for sample_idx in range(x.shape[0])
            ], dim=0)
            for modality_idx in range(self.num_modalities)
        ], dim=1)

        output = output * weight
        output = output.sum(dim=1)

        return output


class DMoMEProbLevel(nn.Module):
    def __init__(self, ignore_assert=False):
        super().__init__()
        if ignore_assert:
            print("CAREFUL!!!!!!YOU ARE IGNORING THE ASSERTATION OF MODEL CREATION!!!!!!")
        else:
            assert not ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'] and not ModelConfig.VAL_LOSS_ARGS['need_sigmoid'], \
                "loss for this model does not need sigmoid!"

        self.pretrained_expert_file_list = ModelConfig.PRETRAINED_EXPERT_FILE_LIST
        self.n_stages = ModelConfig.N_STAGES
        self.num_modalities = ModelConfig.INPUT_CHANNELS
        self.num_classes = ModelConfig.N_CLASSES
        if len(self.pretrained_expert_file_list) != self.num_modalities:
            raise ValueError('PRETRAINED_EXPERT_FILE_LIST must match INPUT_CHANNELS')

        self.expert_ls = nn.ModuleList([
            self._build_single_expert(expert_id)
            for expert_id in range(self.num_modalities)
        ])
        self.router = nn.Sequential(
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=self.num_modalities * 64, out_features=self.num_modalities * self.num_classes),
        ).cuda()

        self.expert_output_shape = self.expert_ls[0](torch.rand(1, 1, 128, 128, 128).cuda()).shape

    def _build_single_expert(self, expert_id):
        net = UNet(
            input_channels=1,
            n_classes=ModelConfig.N_CLASSES,
            n_stages=ModelConfig.N_STAGES,
            n_features_per_stage=ModelConfig.N_FEATURES_PER_STAGE,
            kernel_size=ModelConfig.KERNEL_SIZES,
            strides=ModelConfig.STRIDES
        ).cuda()

        if self.pretrained_expert_file_list[expert_id] is not None:
            net.load_state_dict(torch.load(self.pretrained_expert_file_list[expert_id]))

        return net

    def forward(self, x):
        modality_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)

        weight = self.router(x).view(-1, self.num_modalities, self.num_classes, 1, 1, 1)
        weight[modality_mask] = float('-inf')
        weight = nn.functional.softmax(weight, dim=1)


        output = torch.stack([
            torch.cat([
                torch.zeros(self.expert_output_shape).cuda()
                if modality_mask[sample_idx, modality_idx]
                else torch.sigmoid(
                    self.expert_ls[modality_idx](x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...])
                )
                for sample_idx in range(x.shape[0])
            ], dim=0)
            for modality_idx in range(self.num_modalities)
        ], dim=1)

        output = output * weight

        output = torch.clamp(output.sum(dim=1), min=0.0, max=1.0)

        return output


class ExpertNet(UNet):
    def __init__(self, input_channels, n_classes, n_stages, n_features_per_stage, kernel_size, strides):
        super().__init__(input_channels, n_classes, n_stages, n_features_per_stage, kernel_size, strides)

    def forward(self, x):
        encoded_feat_maps = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoded_feat_maps.append(x)

        low_res_input = encoded_feat_maps[-1]
        for i in range(len(self.decoder_stages)):
            output = self.connect_layers[i](low_res_input)
            output = torch.cat((output, encoded_feat_maps[-i - 2]), dim=1)
            output = self.decoder_stages[i](output)
            low_res_input = output

        return output


class DMoMEFeatureLevel(nn.Module):
    def __init__(self, ignore_assert=False):
        super().__init__()
        if ignore_assert:
            print("CAREFUL!!!!!!YOU ARE IGNORING THE ASSERTATION OF MODEL CREATION!!!!!!")
        else:
            assert ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'] and ModelConfig.VAL_LOSS_ARGS['need_sigmoid'], \
                "loss for this model needs sigmoid!"
        self.pretrained_expert_file_list = ModelConfig.PRETRAINED_EXPERT_FILE_LIST
        self.n_stages = ModelConfig.N_STAGES
        self.num_modalities = ModelConfig.INPUT_CHANNELS
        if len(self.pretrained_expert_file_list) != self.num_modalities:
            raise ValueError('PRETRAINED_EXPERT_FILE_LIST must match INPUT_CHANNELS')

        self.expert_ls = nn.ModuleList([
            self._build_single_expert(expert_id)
            for expert_id in range(self.num_modalities)
        ])
        self.router = nn.Sequential(
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.num_modalities, out_channels=self.num_modalities, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.num_modalities, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=self.num_modalities * 64, out_features=self.num_modalities),
        ).cuda()
        self.segmentation_head = nn.Conv3d(
            in_channels=self.expert_ls[0].seg_layers[-1].in_channels,
            out_channels=self.expert_ls[0].seg_layers[-1].out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.expert_output_shape = self.expert_ls[0](torch.rand(1, 1, 128, 128, 128).cuda()).shape

    def _build_single_expert(self, expert_id):
        net = ExpertNet(
            input_channels=1,
            n_classes=ModelConfig.N_CLASSES,
            n_stages=ModelConfig.N_STAGES,
            n_features_per_stage=ModelConfig.N_FEATURES_PER_STAGE,
            kernel_size=ModelConfig.KERNEL_SIZES,
            strides=ModelConfig.STRIDES
        ).cuda()

        if self.pretrained_expert_file_list[expert_id] is not None:
            net.load_state_dict(torch.load(self.pretrained_expert_file_list[expert_id]))

        return net

    def forward(self, x):
        # True if modality missing
        modality_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)

        weight = self.router(x)
        weight[modality_mask] = float('-inf')
        weight = nn.functional.softmax(weight, dim=1)

        output = torch.stack([
            torch.cat([
                torch.zeros(self.expert_output_shape).cuda()
                if modality_mask[sample_idx, modality_idx]
                else self.expert_ls[modality_idx](x[sample_idx:sample_idx+1, modality_idx:modality_idx+1, ...]
                                                  )*weight[sample_idx, modality_idx]
                for sample_idx in range(x.shape[0])
            ], dim=0)
            for modality_idx in range(self.num_modalities)
        ], dim=1)
        output = output.sum(dim=1)
        output = self.segmentation_head(output)

        return output
