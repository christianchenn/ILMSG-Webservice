import tensorflow as tf
import numpy as np
import random, pickle
from src.utils.video import read_frames
from src.utils.video import stretch_contrast
from src.data.transforms.VideoTransformsV5 import VideoTransformsV5

class VideoDatasetV3:
    def __init__(self, label_dir, video_dir, filenames, transforms):
        self.label_dir = label_dir
        self.video_dir = video_dir
        self.filenames = filenames
        self.transform = transforms

    def __len__(self):
        return len(self.label_dir)

    def __getitem__(self,idx):
        video = self.read_video(idx)
        label = self.read_label(idx)
        # print(video.shape)
        # print(label.shape)
        return video, label
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()
            
    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        np.random.shuffle(self.filenames)
    
    def read_label(self, idx):
        label_path = f"{self.label_dir}/{self.filenames[idx]}.pkl"
        label = None
        with open(label_path, 'rb') as f:
            label = pickle.load(f)
        return tf.constant(label, dtype=tf.float32)
    
    def read_video(self, idx):
        video_path = f"{self.video_dir}/{self.filenames[idx]}.MP4"
        frames, _ = read_frames(video_path, False)
        video = []
        for frame in frames:
            video.append(self.transform(tf.constant(frame, dtype=tf.float32)))
        return tf.stack(video)
    