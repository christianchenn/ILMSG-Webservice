import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from src.utils.logging import log_mel_images, log_mel_audio_table
from src.visualization.audio import visualize_mels


class VideoTestLogAudioPredictionCallback(Callback):

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            x, y = batch
            log_mel_audio_table(
                outputs=outputs,
                batch=y,
                logger=trainer.logger
            )