import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from src.utils.logging import calculate_audio_accuracy, log_accuracy, log_mel_images, log_mel_audio_table
from src.visualization.audio import visualize_mels


class TestLogAudioPredictionCallback(Callback):

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        yaml_file = None
        if pl_module.yaml_file:
            yaml_file = pl_module.yaml_file
        if batch_idx == 0:
            log_mel_audio_table(
                outputs=outputs,
                batch=batch,
                logger=trainer.logger,
                yaml_file=yaml_file
            )
        
        calculate_audio_accuracy(
            audio_model=pl_module,
            y=batch,
            y_hat=outputs,
            yaml_file=yaml_file
        )
    def on_test_end(self, trainer, pl_module):
        log_accuracy(
            video_model=pl_module,
            logger=trainer.logger,
        )
        
        