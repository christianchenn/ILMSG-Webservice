import torch
import pickle
from src.utils.engine import normalize


class AudioDatasetV2(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        filepath = self.paths[idx]
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            data = torch.from_numpy(data)
            data = normalize(data)
            # print("MAX",data.max())
            # print("MIN",data.min())
            return data.unsqueeze(0)

    def item(self, idx, device="cuda"):
        filepath = self.paths[idx]
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            data = torch.from_numpy(data).to(device)
            data = normalize(data)
            
        return data.unsqueeze(0), filepath
