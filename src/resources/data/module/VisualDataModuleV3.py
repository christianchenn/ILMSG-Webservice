from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np
import tensorflow as tf
from src.data.dataset.VideoDatasetV3 import VideoDatasetV3


class VideoDataModuleV3():
    def __init__(self, label_dir, video_dir, batch_size=32, transforms=None):
        self.batch_size = batch_size
        self.label_dir = label_dir
        self.video_dir = video_dir
        self.train_generator = self.val_generator = self.test_generator = None
        self.train_filenames = self.val_filenames = self.test_filenames = []
        self.transforms = transforms

        self.ot = (tf.float32, tf.float32)
        self.os = None

    def prepare_data(self) -> None:
        self.train_filenames = [f'train/{filename.split(".")[0]}' for filename in os.listdir(f"{self.video_dir}/train")]
        self.test_filenames = [f'test/{filename.split(".")[0]}' for filename in os.listdir(f"{self.video_dir}/test")]
        self.val_filenames = [f'val/{filename.split(".")[0]}' for filename in os.listdir(f"{self.video_dir}/val")]

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            self.train_generator = VideoDatasetV3(
                label_dir=self.label_dir,
                video_dir=self.video_dir,
                filenames=self.train_filenames,
                transforms=self.transforms
            )
            self.val_generator = VideoDatasetV3(
                label_dir=self.label_dir,
                video_dir=self.video_dir,
                filenames=self.val_filenames,
                transforms=self.transforms
            )
        if stage == "test":
            self.test_generator = VideoDatasetV3(
                label_dir=self.label_dir,
                video_dir=self.video_dir,
                filenames=self.test_filenames,
                transforms=self.transforms
            )
    
    def update_os(self, input_shape, output_shape):
        self.os = (
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=output_shape, dtype=tf.float32)
            )
            
    def train_dataset(self):
        # print(self.os)
        ds = tf.data.Dataset.from_generator(self.train_generator, output_signature=self.os)
        # ds = ds.prefetch(1)
        ds = ds.batch(self.batch_size)
        return ds

    def val_dataset(self):
        ds = tf.data.Dataset.from_generator(self.val_generator, output_signature=self.os)
        # ds = ds.prefetch(1)
        ds = ds.batch(self.batch_size)
        return ds

    def test_dataset(self):
        ds = tf.data.Dataset.from_generator(self.test_generator, output_signature=self.os)
        # ds = ds.prefetch(1)
        ds = ds.batch(self.batch_size)
        return ds