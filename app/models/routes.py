from app.models import bp
from src.utils.engine import yaml_read_directory, yaml_search
from flask import request, jsonify
import os

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

    if url:
        pass
    else:
        pass