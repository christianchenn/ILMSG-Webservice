import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np

from src.data.dataset.AudioDatasetV1 import AudioDatasetV1


class AudioDataModuleV1(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, test_size=0.3, val_size=0.3):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.train_paths = []
        self.test_paths = []
        self.val_paths = []
        self.train_dataset = []
        self.test_dataset = []
        self.val_dataset = []

    def prepare_data(self) -> None:
        # Split disini
        filepaths = [f"{self.data_dir}/{path}" for path in os.listdir(self.data_dir)]
        train_paths, self.test_paths = train_test_split(filepaths, test_size=self.test_size)
        self.train_paths, self.val_paths = train_test_split(filepaths, test_size=(self.val_size / (1 - self.test_size)))

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            self.train_dataset = AudioDatasetV1(paths=self.train_paths)
            self.val_dataset = AudioDatasetV1(paths=self.val_paths)
        if stage == "test":
            self.test_dataset = AudioDatasetV1(paths=self.test_paths)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)