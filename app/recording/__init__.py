from flask import Blueprint

bp = Blueprint('recording', __name__)

# import counties module routes
from app.recording import routes
