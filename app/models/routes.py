import torch

from app.models import bp
from src.utils.engine import yaml_read_directory, yaml_search, get_recording_paths, read_frames, sew_audio
from flask import request, jsonify
import os

from src.utils.model import get_visual_model, get_audio_model, find_ckpt
from src.utils.transforms import get_video_transforms


@bp.get('/')
def fetch():
    cwd = os.getcwd()
    query = request.args.to_dict()
    page = int(query["page"])
    experiments, total = yaml_read_directory(f"{cwd}/src/experiments/video", page=page)

    return {
        "_meta":{
            "status": 200,
            "message": "Models Returned Successfully"
        },
        "list":experiments,
        "page": page,
        "count": total
    }

@bp.get('/<int:id>')
def get(id):
    cwd = os.getcwd()
    experiment = yaml_search(f"{cwd}/src/experiments/video", id)
    return {
        "_meta":{
            "status": 200,
            "message": "Model Returned Successfully"
        },
        "experiment": experiment
    }

@bp.post("/generate")
def generate():
    cwd = os.getcwd()
    recording = request.form.get("recording")
    url = request.form.get("url")
    run_model = request.form.get("run")
    experiment = yaml_search(f"{cwd}/src/experiments/video", run_model)

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
        label_files = get_recording_paths(recording["id"], label_dir)
        video_files = get_recording_paths(recording["id"], video_dir)

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
        