from app.recording import bp
from src.utils.engine import yaml_search, paginate_csv
from flask import request, jsonify
import os


@bp.get('/')
def fetch():
    cwd = os.getcwd()
    query = request.args.to_dict()
    page = int(query["page"])
    set_type = None
    speaker_name = None
    id = int(query["id"])
    print(query)
    print(query.keys())
    if "filter[datasetType]" in query.keys():
        set_type = query["filter[datasetType]"]
        print(set_type)
    if "speaker_name" in query:
        speaker_name = query["speaker_name"]
    experiment = yaml_search(f"{cwd}/src/resources/experiments/video", id)
    hparams = experiment["hyperparameters"]
    data = experiment["data"]
    note = f"note_{data['gender']}_{data['size']}x_seed-{hparams['seed']}.csv"
    recordings, total = paginate_csv(f"{cwd}/src/resources/data/notes/{note}", page=page, set_type=set_type, speaker_name=speaker_name)

    return {
        "_meta": {
            "status": 200,
            "message": "Recordings Returned Successfully"
        },
        "list": recordings,
        "page": page,
        "count": total
    }