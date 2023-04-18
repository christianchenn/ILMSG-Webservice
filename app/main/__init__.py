from flask import Blueprint

bp = Blueprint('main', __name__)

# import counties module routes
from app.main import routes
