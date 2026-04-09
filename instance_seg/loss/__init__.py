"""Loss functions for instance segmentation."""

from .embedding_loss import DiscriminativeLoss
from .loss_function import (
    segmentation_loss,
    MixSoftmaxCrossEntropyLoss,
    DiceLoss,
    BCELossBoud,
    CustomKLLoss,
    AdaptiveTaskWeighting,
    compute_sdf_loss,
    compute_supervised_task_losses,
    compute_unsupervised_task_losses,
    softmax_mse_loss,
    entropy_loss
)

__all__ = [
    'DiscriminativeLoss',
    'segmentation_loss',
    'MixSoftmaxCrossEntropyLoss',
    'DiceLoss',
    'BCELossBoud',
    'CustomKLLoss',
    'AdaptiveTaskWeighting',
    'compute_sdf_loss',
    'compute_supervised_task_losses',
    'compute_unsupervised_task_losses',
    'softmax_mse_loss',
    'entropy_loss'
]
