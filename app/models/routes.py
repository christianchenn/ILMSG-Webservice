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
    print(id)
    cwd = os.getcwd()
    experiment = yaml_search(f"{cwd}/src/experiments/video", id)
    return {
        "_meta":{
            "status": 200,
            "message": "Model Returned Successfully"
        },
        "experiment": experiment
    }

