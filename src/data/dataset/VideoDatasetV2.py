import torch
import pickle
import re

from src.utils.video import read_frames, stretch_contrast


class VideoDatasetV2(torch.utils.data.Dataset):
    def __init__(self, audio_dir, video_dir, filenames, transforms):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.filenames = filenames
        self.transform = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        video = self.read_video(idx)
        # label = self.read_label(idx)
        audio = self.read_audio(idx)
        return video, audio

    def read_label(self, idx):
        label_path = f"{self.label_dir}/{self.filenames[idx]}.pkl"
        label = None
        with open(label_path, 'rb') as f:
            label = pickle.load(f)
        return label
    
    def read_video(self, idx):
        video_path = f"{self.video_dir}/{self.filenames[idx]}.MP4"
        frames, _ = read_frames(video_path, False)
        video = []
        for frame in frames:
            video.append(self.transform(torch.from_numpy(frame)))
        return torch.stack(video)
    
    def read_audio(self, idx):
        

        filename = self.filenames[idx]
        
        # Use regular expression to extract the numeric range from the filename
        match = re.search(r"(\d+)-(\d+)$", filename)
        start_num, end_num = int(match.group(1)), int(match.group(2))
        # print(start_num, end_num)

        # Multiply the numeric range by 640
        start_num_new, end_num_new = start_num * 640, end_num * 640

        # Construct the new filename with the updated numeric range
        new_filename = re.sub(r"(\d+)-(\d+)$", f"{start_num_new}-{end_num_new}.pkl", filename)

        audio_path = f"{self.audio_dir}/{new_filename}"
        with open(audio_path, 'rb') as f:
            audio = pickle.load(f)
            audio = torch.from_numpy(audio)
            audio = audio.squeeze()

        return audio