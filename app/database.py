# app/database.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    url = 'mariadb+mariadbconnector://admin:admin@localhost/ilmsg'
    app.config["SQLALCHEMY_DATABASE_URI"] = url
    # initialize the app with the extension
    db.init_app(app)