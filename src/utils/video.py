from datetime import datetime
import os
import pickle
from subprocess import call, check_output
import subprocess
import numpy as np
import cv2
# import dlib
# import torch
from tqdm import tqdm
from pathlib import Path
import pathlib as p
from src.utils.engine import split_audio_idx, split_video_idx
import torch

def split_video(input_dir, output_dir, set_type, paths, frame, color, stride=1, validate_only=False):
    output_files = []
    for file_path in tqdm(paths):
        basename = os.path.basename(file_path).split(".")[0]

        # Generate Video Path
        Path(f"{output_dir}/{set_type}/").mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/{set_type}/{basename}"
        _path = f"{input_dir}/{file_path}.MP4"

        outputs = vid_to_batch(
            input_file=_path,
            output_file=output_path,
            split_frames=frame,
            get_path_only=validate_only,
            stride = stride,
            color=color
        )
        output_files += outputs
    return output_files


def convert_framerate(input_file, output_file):
    # ffmpeg -i input.mp4 -vf "minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=ds:fps=50'" output.mp4
    # ffmpeg -i input.mp4 -vf "minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=ds:fps=25'" -an -crf 18 output.mp4
    # ffmpeg -i R17156_S464.MP4 -vf "minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=ds:fps=25'" -an -crf 18 output_optical2.mp4

    command = [
        '/usr/bin/ffmpeg',
        '-i', f"{input_file}",
        '-an',
        '-vf', "minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=ds:fps=25'",
        '-c:v', "libx264",
        '-crf', '20',
        output_file
    ]
    print(" ".join(command))
    try:
        call(command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


def cut_frames(frames, target_frames=82):
    total = len(frames)
    diff = total - target_frames
    if diff == 1:
        return frames[:-1]
    elif diff == 0:
        return frames
    elif diff == 2:
        return frames[1:-1]

    return False

def write_video(path, frames, fps, width, height, color):
    fourCC = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, int(fourCC), fps, (width, height), isColor=color)
    for frame in frames:
        out.write(frame)
    out.release()

def vid_to_batch(input_file, output_file, color=True, stride=1, fps=25, split_frames=25, get_path_only=False):
    outputs = []

    frames, (width, height) = read_frames(input_file, color)
    frames = cut_frames(frames, 82)
    split_idx = split_video_idx(len(frames), split_frames, stride=stride)
    for i, j in split_idx:
        cropped_frames = frames[i:j]
        final_output = f"{output_file}_{i}-{j}.MP4"
        outputs.append(final_output)
        if not get_path_only:
            write_video(final_output, cropped_frames, fps, width, height, color)

    return outputs


def feature_landmarks(landmarks):
    V1 = landmarks[48]
    V2 = np.round((landmarks[50] + landmarks[52]) / 2).astype(int)
    V3 = landmarks[54]
    V4 = landmarks[57]
    return np.array((V1, V2, V3, V4))


def centroid(vertices):
    _len = len(vertices)
    _x = np.round(vertices[:, 0].sum() / _len).astype(int)
    _y = np.round(vertices[:, 1].sum() / _len).astype(int)
    return np.array((_x, _y))


def lip_to_centroid(frames, detector, predictor):
    centroids = []
    for frame in frames:
        faces = detector(frame, 1)
        landmark = predictor(frame, faces[0])
        landmarks = np.array([(p.x, p.y) for p in landmark.parts()])
        # Get 4 features landmarks
        vertices = feature_landmarks(landmarks)
        # Calculate & Get max Distance
        c = centroid(vertices)
        centroids.append(c)
    avg_centroid = np.average(centroids, axis=0)
    return avg_centroid


def vid_to_centroid(filepath, detector, predictor):
    cap = cv2.VideoCapture(filepath)
    fCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    counter = 1
    while (cap.isOpened() and counter <= fCount):
        _, frame = cap.read()
        counter = counter + 1

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(frame)
    cap.release()
    return lip_to_centroid(frames, detector, predictor)


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


def crop_lip(frames, centroid, out, vid_size):
    half_len_w = vid_size[0] / 2
    half_len_h = vid_size[1] / 2
    # Calculate new Vertex
    new_V1 = np.round(np.array([centroid[0] - half_len_w, centroid[1] - half_len_h])).astype(int)
    new_V2 = np.round(np.array([centroid[0] + half_len_w, centroid[1] + half_len_h])).astype(int)
    cropped_frames = []

    for frame in frames:
        frame[new_V1[1]:new_V2[1], new_V1[0]:new_V2[0]]
        # Crop
        cropped = frame[new_V1[1]:new_V2[1], new_V1[0]:new_V2[0]]
        out.write(np.array(cropped))
        cropped_frames.append(cropped)
    return np.array(cropped_frames)


def stretch_contrast(frame, p):
    p1, p2 = p
    rmax = frame.max()
    rmin = frame.min()
    imin, imax = torch.kthvalue(frame.flatten(), int(len(frame.flatten()) * p1 // 100)), torch.kthvalue(frame.flatten(),
                                                                                                        int(len(
                                                                                                            frame.flatten()) * p2 // 100))
    imin, imax = imin[0], imax[0]
    frame = torch.clamp(frame, imin, imax)
    return torch.round((frame - rmin) * ((imax - imin) / (rmax - rmin)) + imin)

from skimage import exposure, io


def stretch_contrast_v2(frame, p):
    p1, p2 = p
    p1, p2 = np.percentile(frame, (p1, p2))
    img_rescale = exposure.rescale_intensity(frame, in_range=(p1, p2))
    return img_rescale


import moviepy.editor as mp
from moviepy.audio.AudioClip import AudioArrayClip
import soundfile as sf
import os


def combine_video(video_path, frames, audio, sr, fps):
    sf.write("./temp.WAV", audio, sr)
    video = mp.ImageSequenceClip(list(frames), fps=fps)
    audio = mp.AudioFileClip("./temp.WAV", fps=sr)

    final_clip = video.set_audio(audio)
    final_clip.set_duration(audio.duration)

    final_clip.write_videofile(video_path, fps=25, audio_codec="aac", codec="libx264")
    os.remove("./temp.WAV")