import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np

from src.data.dataset.AudioDatasetV1 import AudioDatasetV1


class AudioDataModuleV2(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train_dataset = []
        self.test_dataset = []
        self.val_dataset = []

    def prepare_data(self) -> None:
        self.train_paths = [f"{self.data_dir}/train/{filename}" for filename in os.listdir(f"{self.data_dir}/train")]
        self.test_paths = [f"{self.data_dir}/test/{filename}" for filename in os.listdir(f"{self.data_dir}/test")]
        self.val_paths = [f"{self.data_dir}/val/{filename}" for filename in os.listdir(f"{self.data_dir}/val")]

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            self.train_dataset = AudioDatasetV1(paths=self.train_paths)
            self.val_dataset = AudioDatasetV1(paths=self.val_paths)
        if stage == "test":
            self.test_dataset = AudioDatasetV1(paths=self.test_paths)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)