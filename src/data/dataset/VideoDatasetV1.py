import torch
import pickle
import cv2

from src.utils.video import read_frames, stretch_contrast


class VideoDatasetV1(torch.utils.data.Dataset):
    def __init__(self, label_dir, video_dir, filenames, transforms, color=False):
        self.label_dir = label_dir
        self.video_dir = video_dir
        self.filenames = filenames
        self.transform = transforms
        self.color = color
        print("COLOR",self.color)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        video = self.read_video(idx)
        label = self.read_label(idx)
        return video, label

    def read_label(self, idx):
        label_path = f"{self.label_dir}/{self.filenames[idx]}.pkl"
        label = None
        with open(label_path, 'rb') as f:
            label = pickle.load(f)
        return label
    
    def read_video(self, idx):
        video_path = f"{self.video_dir}/{self.filenames[idx]}.MP4"
        frames, _ = read_frames(video_path, self.color)
        video = []
        for frame in frames:
            # if self.color:
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mod_frame = self.transform(torch.from_numpy(frame))
            video.append(mod_frame)
        return torch.stack(video)
    
    # def read_audio(self, idx):
    #     audio_path = f"{self.audio_dir}/{self.filenames[idx]}.MP4"
    #     with open(audio_path, 'rb') as f:
    #         audio = pickle.load(f)
    #     return audio