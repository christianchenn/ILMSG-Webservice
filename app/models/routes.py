import librosa
import scipy
import torch

from app.models import bp
from src.utils.engine import yaml_read_directory, yaml_search, get_recording_paths, read_frames, sew_audio, \
    extract_recording_file, read_label
from flask import request, jsonify
import os

from src.utils.model import get_visual_model, get_audio_model, find_ckpt
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
    experiment = yaml_search(f"{cwd}/src/experiments/video", id)
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
    data = request.values.to_dict()
    if len(request.values.to_dict().keys()) == 0:
        data = request.get_json()
    print(data)
    # recording = data["rid"]
    recording = 3306
    url = None
    if "url" in request.values.to_dict().keys():
        url = data["url"]
    run_model = data["run"]
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

    if url:
        pass
    else:
        processed_dir = f"{cwd}/data/processed"
        arr_size = str(data["frame_size"]).split("x")
        color = "" if not data["color"] else "-color"
        if len(arr_size) == 1:
            arr_size = [int(arr_size[0]), int(arr_size[0])]
        else:
            arr_size = [int(arr_size[0]), int(arr_size[1])]
        video_dir = f"{processed_dir}/{data['size']}x/seed-{seed}/{data['gender']}/video/{arr_size[0]}x{arr_size[1]}{color}/F{data['frames']}"
        label_dir = f"{processed_dir}/{data['size']}x/seed-{seed}/{data['gender']}/label/mels-{data['n_mels']}/{data['audio_run']}/F{data['frames']}"
        label_files = get_recording_paths(recording, label_dir)
        video_files = get_recording_paths(recording, video_dir)
        interim_dir = f"{cwd}/data/interim/{data['gender']}"
        raw_video_dir = f"{interim_dir}/video/raw"
        audio_dir = f"{interim_dir}/audio"
        ori_video_filename = f"{extract_recording_file(video_files[0])}.MP4"
        ori_audio_filename = f"{extract_recording_file(video_files[0])}.WAV"
        ori_audio_path = f"{audio_dir}/{ori_audio_filename}"
        ori_video_path = f"{raw_video_dir}/{ori_video_filename}"
        ori_video, (h, w) = read_frames(ori_video_path, True)

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

        # Prediction
        video_batch = []
        for i, file in enumerate(video_files):
            frames, (h, w) = read_frames(file, False)
            f = []
            for frame in frames:
                f.append(transforms(torch.from_numpy(frame)))
            video_batch.append(torch.stack(f))
        video_batch = torch.stack(video_batch).cuda()
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
        labels = []
        for i, file in enumerate(label_files):
            label = read_label(file)
            labels.append(label)
        labels = torch.stack(labels).cuda()
        latent_mels = audio_model.decoder(labels)
        latent_wav = sew_audio(latent_mels)
        combine_video(
            filepath_latent,
            frames=ori_video,
            audio=latent_wav,
            fps=25,
            sr=16000
        )

        # Original Video
        y, _ = librosa.load(ori_audio_path, sr=16000)
        combine_video(
            filepath_ori,
            frames=ori_video,
            audio=y,
            fps=25,
            sr=16000
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
