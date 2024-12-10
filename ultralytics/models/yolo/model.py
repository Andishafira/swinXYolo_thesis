# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel
import timm
from timm.models import swin_transformer
import torch.nn as nn
from timm import create_model

class YOLO(Model):
    """
    YOLO (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, }

# Definisikan SwinTransformer
class SwinTransformer(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinTransformer, self).__init__()
        # Inisialisasi model dari timm
        self.model = create_model('swin_base_patch4_window7_224', pretrained=pretrained)

    def forward(self, x):
        x = self.model(x)
        return x