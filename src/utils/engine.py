import os
import fnmatch
import yaml
import torch
import pickle
import cv2
import librosa
import soundfile as sf
import pandas as pd
import numpy as np

from src.utils.video import combine_video


def yaml_read_directory(directory_path, count=10, page=0, sort_order='desc'):
    files = os.listdir(directory_path)
    files = [f for f in files if f.endswith('.yaml') and "999" not in f]
    files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    if sort_order == "desc":
        files.reverse()
    total_count = len(files)
    start = (page - 1) * count
    end = start + count
    files_batch = files[start:end]
    data = []
    for file_name in files_batch:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path) as file:
            data.append(yaml.safe_load(file))
    return data, total_count


def paginate_csv(directory_path, count=10, page=0, sort_order='desc', set_type=None, speaker_name=None):
    df = pd.read_csv(directory_path)
    df = df.drop("Unnamed: 0", axis=1)
    if set_type:
        df = df.loc[df['set_type'] == set_type]
    if speaker_name is not None:
        df = df[df['speaker_name'].str.contains(speaker_name)]
    if sort_order == 'desc':
        df = df.sort_values(by=['recordingId'], ascending=False)
    else:
        df = df.sort_values(by=['recordingId'], ascending=True)
    total_count = len(df)
    start = (page - 1) * count
    end = start + count
    data = df.iloc[start:end].to_dict(orient='records')
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
    video_name = "R" + str(rid)
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


def get_recording_filename(rid, directory):
    video_name = "R" + str(rid)
    for file in os.listdir(directory):
        if video_name in file:
            return file
    return None


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
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


def mel_to_wav(mels, audio_path="recreated_audio.wav", gain=35.0, from_db=False, sr=16000, n_iter=64, n_mels=128,
               fmin=0, fmax=None, fps=25,
               save=True):
    mels = mels.squeeze()
    # fmax = fmax if fmax else sr / 2.0
    frame_length = sr // fps
    if from_db:
        mels = librosa.db_to_power(mels)
    wav = librosa.feature.inverse.mel_to_audio(mels, n_iter=64, sr=sr, center=True, n_fft=2048, fmax=sr / 2.0,
                                               hop_length=frame_length)
    wav = np.clip(wav * gain, -1.0, 1.0)

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


def split_video_idx(frame_count, split_frames, stride=1):
    split_idx = []
    for i in range(0, frame_count, stride):
        if (i + split_frames) <= frame_count:
            split_idx.append((i, (i + split_frames)))
    return split_idx


def split_audio_idx(y, frame_length, split_frames, stride=1):
    split_idx = []
    for i in range(0, frame_length, stride):
        if (i + split_frames) * frame_length <= len(y):
            split_idx.append((i * frame_length, (i + split_frames) * frame_length))
    return split_idx


def extract_recording_file(input_string):
    # Split the input string using the '/' separator
    path_parts = input_string.split('/')

    # Get the last part of the path, which is the filename
    filename = path_parts[-1]

    # Split the filename using the '_' separator
    filename_parts = filename.split('_')

    # Extract the desired substring by joining the first two parts of the filename
    desired_substring = '_'.join(filename_parts[:2])

    return desired_substring


def predict_audio():
    pass


def predict(visual_model, video_batch, audio_model, filepaths, ori_video, ori_audio, label_batch = None):
    filepath_prediction, filepath_latent, filepath_ori = filepaths
    # Prediction
    latents = visual_model(video_batch)
    target_mels = audio_model.decoder(latents)
    target_wav = sew_audio(target_mels)
    combine_video(
        filepath_prediction,
        frames=ori_video,
        audio=target_wav,
        fps=25,
        sr=16000
    )

    # Latent Prediction
    if label_batch is not None:
        latent_mels = audio_model.decoder(label_batch)
        latent_wav = sew_audio(latent_mels)
        combine_video(
            filepath_latent,
            frames=ori_video,
            audio=latent_wav,
            fps=25,
            sr=16000
        )

    # Original Video
    combine_video(
        filepath_ori,
        frames=ori_video,
        audio=ori_audio,
        fps=25,
        sr=16000
    )


def predict_with_file():
    pass


def predict_with_files(video_files, label_files, transforms, visual_model, audio_model, ori_video, ori_audio,
                       filepaths):
    # Load Batch Video
    video_batch = []
    for i, file in enumerate(video_files):
        frames, (h, w) = read_frames(file, False)
        f = []
        for frame in frames:
            f.append(transforms(torch.from_numpy(frame)))
        video_batch.append(torch.stack(f))
    video_batch = torch.stack(video_batch).cuda()

    # Load Latent Batch
    labels = []
    for i, file in enumerate(label_files):
        label = read_label(file)
        labels.append(label)
    labels = torch.stack(labels).cuda()

    predict(
        visual_model=visual_model,
        audio_model=audio_model,
        ori_video=ori_video,
        filepaths=filepaths,
        ori_audio=ori_audio,
        label_batch=labels,
        video_batch=video_batch,
    )
