from flask import Flask

from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

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
