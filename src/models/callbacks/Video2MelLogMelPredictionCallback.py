import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from src.utils.logging import log_mel_images
from src.visualization.audio import visualize_mels
from src.utils.audio import mel_to_wav


class Video2MelLogMelPredictionCallback(Callback):

    def on_validation_epoch_end(
            self, trainer, pl_module):
        """Called when the validation epoch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        
        if trainer.current_epoch % 1 == 0:
            # TODO GET 5 DATA THEN LOG
            x, y =  next(iter(trainer.val_dataloaders[0]))
            # print(batch.is_cuda)
            # print(next(pl_module.parameters()).device)
            x = x.cuda()
            y_hat = pl_module(x)
            
            images = []
            audios = []
            gt_audios = []
            captions = []
            
            arr_idx = np.random.randint(0,len(y), size=(2))

            for i in arr_idx:
                caption = f"Data - {(i + 1)}"
                # Log Mel
                captions.append(caption)
                # Change to latent space
                # print(prep_latent(y_hat[i]).size(), "YHAT SIZE")
                # print(prep_latent(y[i]).size(), "Y SIZE")
                _output = y_hat[i]
                
                _input = y[i]
                # print("INPUT:", y[i].unsqueeze(0).max(), y[i].unsqueeze(0).min())
                # print("OUTPUT:", y_hat[i].unsqueeze(0).max(), y_hat[i].unsqueeze(0).min())
                
                # Visualize Mel spectrogram
                img_input = visualize_mels(
                    mels=_input.cpu().squeeze().numpy(),
                    save=True,
                    truth=True,
                    from_db=True
                )
                img_output = visualize_mels(
                    mels=_output.cpu().squeeze().numpy(),
                    save=True,
                    truth=False,
                    from_db=True
                )
                vis = np.concatenate((img_input, img_output), axis=0)
                images.append(vis)
                
                ori_wav = mel_to_wav(
                    mels=_input.cpu().detach().numpy(),
                    save=False,
                    from_db=True
                )
                gt_audios.append(wandb.Audio(data_or_path=ori_wav, caption=caption, sample_rate=16000))
                wav = mel_to_wav(
                    mels=_output.cpu().detach().numpy(),
                    save=False,
                    from_db=True
                )
                audios.append(wandb.Audio(data_or_path=wav, caption=caption, sample_rate=16000))
            
            trainer.logger.log_image(
                key='Validation Images',
                images=images,
                caption=captions)
            wandb.log({"Val Audio": audios})
            wandb.log({"GT Val Audio": gt_audios})