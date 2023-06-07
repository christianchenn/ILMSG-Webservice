import cv2
import librosa
import numpy as np
import pandas as pd
import scipy
import torch
from werkzeug.utils import secure_filename
from datetime import datetime

from app.models import bp
from src.utils.engine import yaml_read_directory, yaml_search, get_recording_paths, read_frames, sew_audio, \
    extract_recording_file, read_label, get_recording_filename, predict_with_files, predict, generate_filenames, generate_videos
from flask import request, jsonify
import os

from src.utils.model import get_visual_model, get_audio_model, find_ckpt
from src.utils.preprocess import preprocess_video, split_video, split_audio, extract_audio, sew_audios, prep_audio
from src.utils.transforms import get_video_transforms
from src.utils.video import combine_video
import shutil

from src.visualization.audio import visualize_mels, visualize_latent


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
    # cwd = f"{os.getcwd()}/src/resources"
    cwd = f"./src/resources"
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
    print(experiment)

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
    (filename_prediction, filepath_prediction), (filename_latent, filepath_latent), (
    filename_ori, filepath_ori) = generate_filenames(cwd)

    arr_size = str(data["frame_size"]).split("x")
    color = "" if not data["color"] else "-color"
    if len(arr_size) == 1:
        arr_size = [int(arr_size[0]), int(arr_size[0])]
    else:
        arr_size = [int(arr_size[0]), int(arr_size[1])]

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
    target_latents = None
    target_mels = None
    label_batch = None
    
    if url is not None:
        pass
    elif 'file' in request.files:
        video_file = request.files['file']
        filename = secure_filename(video_file.filename)
        print(filename)
        temp_ori_video = f"{cwd}/temp/{filename}"
        basename = filename.split(".")[0]
        temp_ori_audio = f"{cwd}/temp/{basename}.WAV"
        print(temp_ori_video)
        print(video_file)
        video_file.save(temp_ori_video)
        ori_video, (h, w) = read_frames(temp_ori_video, True)

        preprocessed_frames = preprocess_video(
            rid=None,
            transforms=transforms,
            frames=ori_video,
            vid_size=arr_size,
            local=False,
            to_gray=True
        )

        extract_audio(
            input_file=temp_ori_video,
            output_file=temp_ori_audio
        )
        ori_audio, _ = librosa.load(temp_ori_video, sr=16000)

        # # Split Video
        video_batch = split_video(
            frames=preprocessed_frames,
            split_frames=data["frames"],
            stride=data["frames"],
            total_frames=len(preprocessed_frames)
        )
        video_batch = video_batch.cuda()

        pred_audios = []
        pred_latents = []
        pred_mels = []
        for j in range(len(video_batch)):
            video = video_batch[j].unsqueeze(0).cuda()

            (ori_mels, ori_latent, ori_audio), (target_mels, target_latent, target_wav) = predict(
                visual_model=visual_model,
                filepaths=(filepath_prediction, filepath_latent, filepath_ori),
                ori_video=ori_video,
                video_batch=video,
                audio_model=audio_model,
                ori_audio=ori_audio,
                label_batch=None,
                generate_video=False
            )
            pred_audios.append(target_wav)
            pred_latents.append(target_latent)
            pred_mels.append(target_mels)

            # audio_model, visual_model = reload_model(cwd, model_conf, data, experiment, learning_rate)
            del video
            torch.cuda.empty_cache()

        target_wav = sew_audios(pred_audios)
        target_wav = prep_audio(target_wav, len(ori_audio))
        target_wav = torch.from_numpy(target_wav)
        ori_audio = torch.from_numpy(ori_audio)

        generate_videos(
            filepaths=(filepath_prediction, filepath_latent, filepath_ori),
            ori_video = ori_video, 
            target_wav = target_wav, 
            ori_audio = ori_audio, 
        )
        
        target_mels = pred_mels
        target_latents = pred_latents
        # os.remove(temp_ori_video)
        # os.remove(temp_ori_audio)

    elif rid is not None:
        recordings = pd.read_json(f"{cwd}/config/recordings.json")
        recording = recordings[recordings["rid"] == int(rid)]
        speakers = pd.read_csv(f"{cwd}/config/speakers.csv", delimiter=";")
        speaker = speakers[speakers["id"] == recording["spid"].item()]
        gender = "pria" if speaker['gender'].item() == "L" else "wanita"

        processed_dir = f"{cwd}/data/processed"
        interim_dir = f"{cwd}/data/interim/{gender}"
        raw_video_dir = f"{interim_dir}/video/raw"
        audio_dir = f"{interim_dir}/audio"

        print(raw_video_dir)
        recording_filename = get_recording_filename(rid, raw_video_dir).split(".")[0]
        ori_video_filename = f"{recording_filename}.MP4"
        ori_audio_filename = f"{recording_filename}.WAV"

        ori_audio_path = f"{audio_dir}/{ori_audio_filename}"
        ori_audio, _ = librosa.load(ori_audio_path, sr=16000)

        ori_video_path = f"{raw_video_dir}/{ori_video_filename}"
        ori_video, (h, w) = read_frames(ori_video_path, True)

        video_dir = f"{processed_dir}/{data['size']}x/seed-{seed}/{gender}/video/{arr_size[0]}x{arr_size[1]}{color}/F{data['frames']}"
        label_dir = f"{processed_dir}/{data['size']}x/seed-{seed}/{gender}/label/mels-{data['n_mels']}/{data['audio_run']}/F{data['frames']}"
        video_files, label_files = [], []
        if os.path.isdir(video_dir) and os.path.isdir(label_dir):
            label_files = get_recording_paths(rid, label_dir)
            video_files = get_recording_paths(rid, video_dir)

        if len(video_files) > 0:
            (latent_mels, label_batch, ori_audio), (target_mels, target_latents, target_wav) = predict_with_files(
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

            (latent_mels, label_batch, ori_audio), (target_mels, target_latents, target_wav) = predict(
                visual_model=visual_model,
                filepaths=(filepath_prediction, filepath_latent, filepath_ori),
                ori_video=ori_video,
                video_batch=video_batch,
                audio_model=audio_model,
                ori_audio=ori_audio,
                label_batch=label_batch
            )
            
    
    has_label = True if label_batch is not None else False
    concat_img_target_mel = []
    concat_img_target_latent = []
    concat_img_input_mel = []
    concat_img_input_latent = []

    for i in range(len(target_mels)):
        target_mel = target_mels[i]
        target_latent = target_latents[i]

        img_target_mel = visualize_mels(
            mels=target_mel.cpu().squeeze().numpy(),
            save=True,
            truth=False,
            from_db=True
        )
        
        concat_img_target_mel.append(img_target_mel)
        
        
        if has_label:
            input_latent = label_batch[i] if label_batch is not None else None
            input_mel = latent_mels[i] if label_batch is not None else None
            img_input_mel = visualize_mels(
                mels=input_mel.cpu().squeeze().numpy(),
                save=True,
                truth=True,
                from_db=True
            )
            img_input_latent = visualize_latent(
                input_latent,
                gt=True,
                save=True,
                vmin= input_latent.min(),
                vmax= input_latent.max(),
            )
            img_target_latent = visualize_latent(
                target_latent,
                gt=False,
                save=True,
                vmin= input_latent.min(),
                vmax= input_latent.max(),
            )
            concat_img_target_latent.append(img_target_latent)
            concat_img_input_mel.append(img_input_mel)
            concat_img_input_latent.append(img_input_latent)
        else:
            img_target_latent = visualize_latent(
                target_latent,
                gt=False,
                save=True,
                vmin= target_latent.min(),
                vmax= target_latent.max(),
            )
            concat_img_target_latent.append(img_target_latent)
        
    def save_image(images, path):
        from PIL import Image
        print(np.array(images).shape)
        image = Image.fromarray(images)
        image.save(path)
    
    def generate_image_filename(desc):
        dt = datetime.now()
        time = int(dt.strftime("%Y%m%d%H%M%S"))
        filepath = f"{cwd}/results"
        filename = f"{time}"
        filename = f"{filename}_{desc}.png"
        filepath = f"{filepath}/{filename}"
        return filepath, filename
        
        
    (target_mels_path, target_mels_filename) = generate_image_filename("Target Mels")
    (target_latent_path, target_latent_filename) = generate_image_filename("Target Latent")
    
    concat_img_target_mel = np.concatenate(concat_img_target_mel, axis=1)
    concat_img_target_latent = np.concatenate(concat_img_target_latent, axis=1)
    save_image(concat_img_target_mel, target_mels_path)
    save_image(concat_img_target_latent, target_latent_path)
    
    input_mels_path = None
    input_latent_path = None
    input_mels_filename = None
    input_latent_filename = None
    if has_label:
        (input_mels_path, input_mels_filename) = generate_image_filename("Input Mels")
        (input_latent_path, input_latent_filename) = generate_image_filename("Input Latent")
        
        concat_img_input_mel = np.concatenate(concat_img_input_mel, axis=1)
        concat_img_input_latent = np.concatenate(concat_img_input_latent, axis=1)
        save_image(concat_img_input_mel, input_mels_path)
        save_image(concat_img_input_latent, input_latent_path)

    return {
        "_meta": {
            "status": 200,
            "message": "Predictions Returned Successfully"
        },
        "data": {
            "original": filename_ori,
            "latent": filename_latent,
            "prediction": filename_prediction,
            "target_mel": target_mels_filename,
            "target_latent": target_latent_filename,
            "input_mel": input_mels_filename,
            "input_latent": input_latent_filename,
        }
    }
