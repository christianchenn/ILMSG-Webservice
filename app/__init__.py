import os

from flask import Flask, send_from_directory
from flask_cors import CORS

from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    # CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
    app.config['CORS_HEADERS'] = 'Content-Type'

    # cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:5173"}})
    cors = CORS(app, resources={r"/models/*": {"origins": "*"}})

    # Initialize Flask extensions here

    # Register blueprints here
    from app.main import bp as main_bp
    from app.models import bp as models_bp
    app.register_blueprint(main_bp, url_prefix='/main')
    app.register_blueprint(models_bp, url_prefix='/models')

    # app/__init__.py
    from app.database import init_db
    init_db(app)

    @app.route('/results/<path:filename>')
    def download_file(filename):
        directory = os.path.join(app.root_path, '../src/resources/results')
        return send_from_directory(directory,
                                   filename, as_attachment=True)

    @app.route('/data/<string:gender>/<string:data_type>/<path:filename>')
    def download_data_file(gender, data_type, filename):
        data_dir = f'../src/resources/data/interim/{gender}/{data_type}'
        if data_type == "video":
            data_dir = f'{data_dir}/raw'
        directory = os.path.join(app.root_path, data_dir)
        return send_from_directory(directory,
                                   filename, as_attachment=True)

    @app.route('/test/')
    def test_page():
        return '<h1>Testing the Flask Application Factory Pattern</h1>'

    return app
