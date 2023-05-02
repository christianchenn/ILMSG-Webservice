import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np

from src.data.dataset.VideoDatasetV1 import VideoDatasetV1


class VideoDataModuleV1(pl.LightningDataModule):
    def __init__(self, label_dir, video_dir, color=False, batch_size=32, num_workers=2, transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.label_dir = label_dir
        self.video_dir = video_dir
        self.train_dataset = self.test_dataset = self.val_dataset = []
        self.train_filenames = self.val_filenames = self.test_filenames = []
        self.transforms = transforms
        self.num_workers = num_workers
        self.color = color

    def prepare_data(self) -> None:
        self.train_filenames = [f'train/{filename.split(".")[0]}' for filename in os.listdir(f"{self.video_dir}/train")]
        self.test_filenames = [f'test/{filename.split(".")[0]}' for filename in os.listdir(f"{self.video_dir}/test")]
        self.val_filenames = [f'val/{filename.split(".")[0]}' for filename in os.listdir(f"{self.video_dir}/val")]

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            self.train_dataset = VideoDatasetV1(
                label_dir=self.label_dir,
                video_dir=self.video_dir,
                filenames=self.train_filenames,
                transforms=self.transforms,
                color=self.color
            )
            self.val_dataset = VideoDatasetV1(
                label_dir=self.label_dir,
                video_dir=self.video_dir,
                filenames=self.val_filenames,
                transforms=self.transforms,
                color=self.color
            )
        if stage == "test":
            self.test_dataset = VideoDatasetV1(
                label_dir=self.label_dir,
                video_dir=self.video_dir,
                filenames=self.test_filenames,
                transforms=self.transforms,
                color=self.color
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
