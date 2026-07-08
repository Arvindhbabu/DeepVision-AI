"""
DeepVision AI

Model Factory
"""

from src.models.vit_temporal_pooling import ViTTemporalPooling
from src.models.efficientnet_bilstm import EfficientNetBiLSTM


class ModelFactory:

    @staticmethod
    def create(config):

        model_name = config["model"]["name"].lower()

        
        if model_name == "vit_temporal_pooling":

            return ViTTemporalPooling(
                image_size=config["dataset"]["image_size"],
                num_classes=config["model"]["num_classes"],
                pretrained=config["model"]["pretrained"],
                freeze_backbone=config["model"]["freeze_backbone"],
                pooling=config["model"]["pooling"],
                hidden_dim=config["model"]["hidden_dim"],
                dropout=config["model"]["dropout"],
        )

        elif model_name == "efficientnet_bilstm":

            return EfficientNetBiLSTM()

        raise ValueError(
            f"Unsupported model: {model_name}"
        )