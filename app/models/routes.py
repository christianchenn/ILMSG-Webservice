import cv2
import librosa
import numpy as np
import pandas as pd
import scipy
import torch
from werkzeug.utils import secure_filename

from app.models import bp
from src.utils.engine import yaml_read_directory, yaml_search, get_recording_paths, read_frames, sew_audio, \
    extract_recording_file, read_label, get_recording_filename, predict_with_files, predict
from flask import request, jsonify
import os

from src.utils.model import get_visual_model, get_audio_model, find_ckpt
from src.utils.preprocess import preprocess_video, split_video, split_audio
from src.utils.transforms import get_video_transforms
from src.utils.video import combine_video
from datetime import datetime
import shutil


@bp.get('/')
def fetch():
    cwd = os.getcwd()
    query = request.args.to_dict()
    page = int(query["page"])
    experiments, total = yaml_read_directory(f"{cwd}/src/resources/experiments/video", page=page)

    return {
        "_meta": {
            "status": 200,
            "message": "Models Returned Successfully"
        },
        "list": experiments,
        "page": page,
        "count": total
    }


@bp.get('/<int:id>')
def get(id):
    cwd = os.getcwd()
    experiment = yaml_search(f"{cwd}/src/resources/experiments/video", id)
    return {
        "_meta": {
            "status": 200,
            "message": "Model Returned Successfully"
        },
        "experiment": experiment
    }


@bp.post("/generate")
def generate():
    cwd = f"{os.getcwd()}/src/resources"
    form = request.form.to_dict()
    print(form)
    print(request.files)

    rid = None
    url = None
    if "url" in request.form.to_dict().keys() and form["url"] != "":
        url = form["url"]
    elif "rid" in request.form.to_dict().keys() and form["rid"] != "":
        rid = form["rid"]

    run_model = form["run"]
    experiment = yaml_search(f"{cwd}/experiments/video", run_model)

    hparams = experiment["hyperparameters"]
    config = experiment["config"]
    data = experiment["data"]

    seed, batch_size, learning_rate = hparams["seed"], hparams["batch_size"], float(hparams["learning_rate"])
    model_conf = experiment["model"]
    config = experiment["config"]
    data = experiment["data"]
    transforms = get_video_transforms(data["transform"], data["color"])

    visual_model = get_visual_model(model_conf["version"], learning_rate, model_conf["name"], experiment)
    audio_model = get_audio_model(data["audio_version"], learning_rate, data["audio_run"])

    # Filenames for Target
    dt = datetime.now()
    time = int(dt.strftime("%Y%m%d%H%M%S"))
    filepath = f"{cwd}/results"
    filename = f"{time}"
    filename_prediction = f"{filename}_Prediction.MP4"
    filepath_prediction = f"{filepath}/{filename_prediction}"
    filename_latent = f"{filename}_Latent.MP4"
    filepath_latent = f"{filepath}/{filename_latent}"
    filename_ori = f"{filename}_Original.MP4"
    filepath_ori = f"{filepath}/{filename_ori}"

    if visual_model == None or audio_model == None:
        raise Exception("No Model Found")
    else:
        def load_model(model, cwd, _type, frames, run_name):
            default_ckpt_path = f"{cwd}/models/ilmsg-{_type}/f{frames}"
            ckpt_filename = find_ckpt(f"{default_ckpt_path}/{run_name}/")
            ckpt_path = f"{default_ckpt_path}/{run_name}/{ckpt_filename}"
            model = model.load_from_checkpoint(ckpt_path).cuda()
            return model

        # Load Audio CKPT
        audio_model = load_model(audio_model, cwd, "audio", data["frames"], data["audio_run"])
        # Load Video CKPT
        visual_model = load_model(visual_model, cwd, "video", data["frames"], model_conf["name"])

    # 3 Types of Input
    if url is not None:
        pass
    elif 'file' in request.files:
        video_file = request.files['file']
        filename = secure_filename(video_file.filename)
        print(filename)
        video_file.save(filename)
        ori_video, (h, w) = read_frames(filename, True)
        os.remove(filename)

    elif rid is not None:
        recordings = pd.read_json(f"{cwd}/resources/config/recordings.json")
        recording = recordings[recordings["rid"] == rid]
        speakers = pd.read_json(f"{cwd}/resources/config/speakers.csv")
        speaker = speakers[speakers["id"] == recording["spid"].item()]
        gender = "pria" if speaker['gender'] == "L" else "wanita"

        processed_dir = f"{cwd}/data/processed"
        interim_dir = f"{cwd}/data/interim/{data['gender']}"
        raw_video_dir = f"{interim_dir}/video/raw"
        audio_dir = f"{interim_dir}/audio"

        arr_size = str(data["frame_size"]).split("x")
        color = "" if not data["color"] else "-color"
        if len(arr_size) == 1:
            arr_size = [int(arr_size[0]), int(arr_size[0])]
        else:
            arr_size = [int(arr_size[0]), int(arr_size[1])]

        print(raw_video_dir)
        recording_filename = get_recording_filename(rid, raw_video_dir).split(".")[0]
        ori_video_filename = f"{recording_filename}.MP4"
        ori_audio_filename = f"{recording_filename}.WAV"

        ori_audio_path = f"{audio_dir}/{ori_audio_filename}"
        ori_audio, _ = librosa.load(ori_audio_path, sr=16000)

        ori_video_path = f"{raw_video_dir}/{ori_video_filename}"
        ori_video, (h, w) = read_frames(ori_video_path, True)

        video_dir = f"{processed_dir}/{data['size']}x/seed-{seed}/{data['gender']}/video/{arr_size[0]}x{arr_size[1]}{color}/F{data['frames']}"
        label_dir = f"{processed_dir}/{data['size']}x/seed-{seed}/{data['gender']}/label/mels-{data['n_mels']}/{data['audio_run']}/F{data['frames']}"
        label_files = get_recording_paths(rid, label_dir)
        video_files = get_recording_paths(rid, video_dir)

        if len(video_files) > 0:
            predict_with_files(
                video_files=video_files,
                filepaths=(filepath_prediction, filepath_latent, filepath_ori),
                transforms=transforms,
                ori_video=ori_video,
                label_files=label_files,
                audio_model=audio_model,
                visual_model=visual_model,
                ori_audio=ori_audio
            )
        else:
            # Preprocess Video
            preprocessed_frames = preprocess_video(
                rid=rid,
                transforms=transforms,
                frames=ori_video,
                vid_size=arr_size,
                local=True,
            )
            # Split Video
            video_batch = split_video(
                frames=preprocessed_frames,
                split_frames=data["frames"],
                stride=data["frames"],
            ).cuda()

            # Predict Target for Validation
            label_batch = split_audio(
                y=ori_audio,
                split_frames=data["frames"],
                stride=data["frames"],
                fps=25,
                sr=16000
            ).cuda()
            label_batch = label_batch.unsqueeze(1)
            label_batch = audio_model.encoder(label_batch)

            predict(
                visual_model=visual_model,
                filepaths=(filepath_prediction, filepath_latent, filepath_ori),
                ori_video=ori_video,
                video_batch=video_batch,
                audio_model=audio_model,
                ori_audio=ori_audio,
                label_batch=label_batch
            )


        return {
            "_meta": {
                "status": 200,
                "message": "Predictions Returned Successfully"
            },
            "data":{
                "original": filename_ori,
                "latent": filename_latent,
                "prediction": filename_prediction
            }
        }
