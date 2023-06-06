import os
import subprocess
from subprocess import call

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from moviepy.editor import VideoFileClip
from src.utils.video import read_frames, cut_frames, crop_lip, lip_to_centroid


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


def pad_wav(y, total_frames, frame_length):
    target_y = total_frames * frame_length
    diff = target_y - len(y)

    # 52480 - 52400
    if diff > 0:
        fill = diff // 2
        return np.pad(y, (fill, fill), 'constant', constant_values=(0, 0))
    return y


def split_audio(y, sr, fps, split_frames=20, stride=None, total_frames=82):
    frame_length = sr // fps
    if stride == None:
        stride = split_frames
    y = pad_wav(y, total_frames, frame_length)
    split_idx = split_audio_idx(y, frame_length, split_frames, stride=stride)
    audio_batch = []
    for i, j in split_idx:
        y_frames = y[i:j]
        S = librosa.feature.melspectrogram(y=y_frames, sr=sr, n_mels=128, fmax=sr / 2.0,
                                           hop_length=frame_length)
        S = librosa.power_to_db(S, ref=np.max)
        audio_batch.append(torch.from_numpy(S))
    return torch.stack(audio_batch)


def split_video(frames, split_frames=20, stride=None, total_frames=82):
    video_batch = []
    frames = cut_frames(frames, total_frames)
    if stride == None:
        stride = split_frames
    split_idx = split_video_idx(len(frames), split_frames, stride=stride)
    for i, j in split_idx:
        trim_frames = frames[i:j]
        video_batch.append(torch.stack(trim_frames))
    return torch.stack(video_batch)


def preprocess_video(frames, rid, vid_size, transforms, local=True, to_gray=True):
    # Convert to Grayscale
    gray_frames = []
    if to_gray:
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)

    # Crop (Dlib)
    centroid = None
    if local:
        def search_centroid(centroids, rid):
            print(f"R{rid}")
            centroid = centroids[centroids['id'] == rid]
            print(centroid["filename"].count())
            if centroid["filename"].count() < 1:
                return None
            return centroid

        centroids_pria = pd.read_csv(f"{os.getcwd()}/src/resources/config/centroids-pria.csv")
        centroids_wanita = pd.read_csv(f"{os.getcwd()}/src/resources/config/centroids-wanita.csv")
        centroid = search_centroid(centroids_pria, rid)
        if centroid is None:
            centroid = search_centroid(centroids_wanita, rid)

    if centroid is None:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(f"{os.getcwd()}/../../src/resources/config/shape_predictor_68_face_landmarks.dat")
        # predictor = dlib.shape_predictor(f"{os.getcwd()}/src/resources/config/shape_predictor_68_face_landmarks.dat")
        print(detector, predictor)
        # Calculate Centroid
        centroid = lip_to_centroid(
            frames=gray_frames,
            detector=detector,
            predictor=predictor
        )
        print(centroid)
        centroid = {
            "cx": centroid[0],
            "cy": centroid[1]
        }

    cropped_frames = crop_lip(
        frames=gray_frames,
        centroid=[float(centroid["cx"]), float(centroid["cy"])],
        vid_size=vid_size
    )

    # Transforms
    preprocessed_frames = []
    for frame in cropped_frames:
        preprocessed_frames.append(transforms(torch.from_numpy(frame)))

    return preprocessed_frames


def convert_to_wav(input_file, output_file, sample_rate=48000, filename="file", highpass=100, lowpass=2500):
    command = ['ffmpeg',
               '-i', input_file,
               '-vn',
               '-af', 'afftdn=nt=w:nf=-30, loudnorm=I=-16:LRA=11:TP=-1, highpass=f=100, lowpass=f=2500',
               '-c:a', 'pcm_s16le',
               output_file
               ]
    print(" ".join(command))
    try:
        call(command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def cvt_25fps(input_file, output_file):
    # Load the video clip
    clip = VideoFileClip(input_file)
    # Set the target frame rate to 25 FPS
    target_fps = 25
    # Convert the clip to 25 FPS
    clip_25fps = clip.set_fps(target_fps)
    # Write the converted clip to a new file
    clip_25fps.write_videofile(output_file, codec="libx264")
    # Close the clip
    clip.close()


def extract_audio(input_file, output_file):
    # Load the video clip
    clip = VideoFileClip(input_file)
    # Extract the audio from the video
    audio = clip.audio
    # Save the audio as a WAV file
    audio.write_audiofile(output_file, codec="pcm_s16le")
    # Close the clip
    clip.close()

def cut_ori_audio(ori_audio, total_frames, sr=16000, fps=25):
    frame = sr//fps
    print(int(total_frames*frame))
    return ori_audio[:int(total_frames*frame)]

def prep_audio(modified_audio, target_length):
    # Adjust the length of the resampled modified audio to match the target length
    if len(modified_audio) > target_length:
        modified_audio = modified_audio[:target_length]
    elif len(modified_audio) < target_length:
        modified_audio = librosa.util.fix_length(modified_audio, size=target_length)

    return modified_audio

def sew_audios(wavs):
    audio_data = []

    for wav in wavs:
        audio_data.append(wav)

    # Concatenate the audio data
    concatenated_data = np.concatenate(audio_data)
    return concatenated_data

def prep_audio(modified_audio, target_length):
    # Adjust the length of the resampled modified audio to match the target length
    if len(modified_audio) > target_length:
        modified_audio = modified_audio[:target_length]
    elif len(modified_audio) < target_length:
        modified_audio = librosa.util.fix_length(modified_audio, size=target_length)

    return modified_audio