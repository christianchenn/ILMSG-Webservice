from flask import Blueprint

bp = Blueprint('models', __name__)

# import counties module routes
from app.models import routes
