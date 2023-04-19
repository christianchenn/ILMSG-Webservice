from flask import Flask
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

    @app.route('/test/')
    def test_page():
        return '<h1>Testing the Flask Application Factory Pattern</h1>'

    return app
