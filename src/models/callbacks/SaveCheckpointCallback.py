from src.utils.upload import authorize_drive, search_folder_id, upload_file, create_folder
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback
import numpy as np

from src.utils.logging import log_mel_images, log_mel_audio_table
from src.visualization.audio import visualize_mels


class SaveCheckpointCallback(Callback):

    def on_save_checkpoint(
            self, trainer, pl_module, checkpoint):

        
        