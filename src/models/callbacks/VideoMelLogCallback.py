import os
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from src.utils.logging import log_latent_audio, log_latent_images
    

class VideoMelLogCallback(Callback):

    def on_validation_epoch_end(
            self, trainer, pl_module):
        """Called when the validation epoch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if trainer.current_epoch % 1 == 0:
            x, y =  next(iter(trainer.val_dataloaders[0]))
            x = x.cuda()
            y = y.cuda()
            y_hat = pl_module(x)
            
            log_latent_images(
                y=y,
                y_hat=y_hat,
                logger=trainer.logger,
                yaml_file=pl_module.yaml_file,
                _type="Validation"
            )
    
    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        
        if batch_idx == 0:
            x, y = batch
            y = y.cuda()
            y_hat = outputs.permute(0, 2, 1, 3)
            y_hat = y_hat.cuda()
            log_latent_audio(
                y=y,
                y_hat=y_hat,
                logger=trainer.logger,
                yaml_file=pl_module.yaml_file,
                _type="Testing"
            )