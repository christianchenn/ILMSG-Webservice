import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from src.utils.logging import log_mel_images
from src.visualization.audio import visualize_mels


class ValLogMelPredictionCallback(Callback):

    def on_validation_epoch_end(
            self, trainer, pl_module):
        """Called when the validation epoch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        
        if trainer.current_epoch % 5 == 0:
            batch =  next(iter(trainer.val_dataloaders[0]))
            # print(batch.is_cuda)
            # print(next(pl_module.parameters()).device)
            batch = batch.cuda()
            # print(batch.size())
            z_hat = pl_module(batch)
            outputs = pl_module.decoder(z_hat)
            yaml_file = None
            if pl_module.yaml_file:
                yaml_file = pl_module.yaml_file
            log_mel_images(
                outputs=outputs,
                batch=batch,
                logger=trainer.logger,
                yaml_file=yaml_file
            )