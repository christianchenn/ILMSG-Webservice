import os
import fnmatch
import yaml
import torch
import pickle
import cv2
import librosa
import soundfile as sf
import numpy as np

def yaml_read_directory(directory_path, count=10, page=0, sort_order='desc'):
    files = os.listdir(directory_path)
    files = [f for f in files if f.endswith('.yaml')]
    files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    if sort_order == "desc":
        files.reverse()
    total_count = len(files)
    start = (page-1) * count
    end = start + count
    files_batch = files[start:end]
    data = []
    for file_name in files_batch:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path) as file:
            data.append(yaml.safe_load(file))
    return data, total_count


def yaml_search(directory, search_number):
    """
    Searches for a YAML file in the specified directory that matches the pattern run_x_.yaml,
    where x is the search number.
    Returns the first matching YAML file as a dictionary.
    """
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, f"run_{search_number}.yaml"):
            with open(os.path.join(directory, filename), 'r') as file:
                return yaml.safe_load(file)
    return None



def get_recording_paths(rid, directory):
    video_name = "R"+str(rid)
    mel_frames = []
    filepaths = []
    _types = ["train", "val", "test"]
    found = False
    for set_type in _types:
        set_dir = f"{directory}/{set_type}"
        for filepath in os.listdir(set_dir):
            if str(video_name) in filepath:
                found = True
                filepaths.append(f"{set_dir}/{filepath}")
        if found:
            sorted_list = sorted(filepaths, key=lambda x: int(x.split('_')[-1].split('.')[0].split("-")[0]))
            return sorted_list
    return []


def read_frames(filepath, color=True):
    cap = cv2.VideoCapture(filepath)
    fCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    counter = 1
    while (cap.isOpened() and counter <= fCount):
        _, frame = cap.read()
        counter = counter + 1
        if not color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    return np.array(frames), (width, height)


def read_label(label_path):
    label = None
    with open(label_path, 'rb') as f:
        label = pickle.load(f)
    return torch.from_numpy(label)


def read_video(video_path, color, transform):
    frames, _ = read_frames(video_path, color)
    video = []
    for frame in frames:
        video.append(transform(torch.from_numpy(frame)))
    return torch.stack(video)


def read_audio(audio_path):
    with open(audio_path, 'rb') as f:
        audio = pickle.load(f)
    return audio

def mel_to_wav(mels, audio_path="recreated_audio.wav", gain=35.0, from_db=False, sr=16000, n_iter=64, n_mels=128, fmin=0, fmax=None, fps=25,
               save=True):
    mels = mels.squeeze()
    # fmax = fmax if fmax else sr / 2.0
    frame_length = sr // fps
    if from_db:
        mels = librosa.db_to_power(mels)
    wav = librosa.feature.inverse.mel_to_audio(mels, n_iter=64, sr=sr, center=True, n_fft=2048, fmax=sr/2.0, hop_length=frame_length)
    wav = np.clip(wav*gain, -1.0, 1.0)

    if save:
        sf.write(audio_path, wav, samplerate=sr)
    else:
        return wav

def sew_audio(list_mels):
    wavs = []
    for mels in list_mels:
        mels = mels.cpu().detach().numpy()
        wav = mel_to_wav(
            mels=mels,
            save=False,
            from_db=True
        )

        wavs.append(wav)
    wavs = np.concatenate(wavs)
    return wavs