from monai.losses import DiceLoss
from torch.nn import Module, BCELoss
from torch import Tensor, sigmoid, clamp, numel


# Regular medical segmentation task loss (Dice loss + BCE loss) + our MoFe loss
class BlendedLoss(Module):
    def __init__(
            self, need_sigmoid=True, compute_ranking_loss=False,
            seg_loss_weight=1, ranking_loss_weight=0
    ):
        super().__init__()
        self.dice = DiceLoss(sigmoid=False, reduction='none')
        self.bce = BCELoss(reduction='none')
        self.need_sigmoid = need_sigmoid
        self.compute_ranking_loss = compute_ranking_loss
        self.seg_loss_weight = seg_loss_weight
        self.ranking_loss_weight = ranking_loss_weight

    def forward(self, preds: Tensor, labels: Tensor):
        """Compute Dice Loss & Binary Cross Entropy

        Args:
            preds (Tensor): [B, C, ...]
            labels (Tensor): [B, C, ...]
        """
        if self.need_sigmoid:
            probs = sigmoid(preds)
        else:
            probs = preds

        dice_loss = self.dice(probs, labels)
        bce_loss = self.bce(probs, labels.float())

        if self.compute_ranking_loss:
            dice_loss = dice_loss.mean(dim=[_ for _ in range(2, len(dice_loss.shape))])
            bce_loss = bce_loss.mean(dim=[_ for _ in range(2, len(bce_loss.shape))])

            dice_loss_more = dice_loss[:dice_loss.shape[0] // 2]
            dice_loss_less = dice_loss[dice_loss.shape[0] // 2:]
            bce_loss_more = bce_loss[:bce_loss.shape[0] // 2]
            bce_loss_less = bce_loss[bce_loss.shape[0] // 2:]

            dice_diff = dice_loss_more - dice_loss_less
            dice_diff = clamp(dice_diff, min=0)
            bce_diff = bce_loss_more - bce_loss_less
            bce_diff = clamp(bce_diff, min=0)
            ranking_loss = dice_diff.mean() + bce_diff.mean()

            seg_loss = dice_loss.mean() + bce_loss.mean()

            loss_dict = {
                'task_loss': seg_loss * self.seg_loss_weight + ranking_loss * self.ranking_loss_weight,
                'seg_loss': seg_loss,
                'ranking_loss': ranking_loss
            }
        else:
            loss_dict = {
                'task_loss': dice_loss.mean() + bce_loss.mean()
            }

        return loss_dict


