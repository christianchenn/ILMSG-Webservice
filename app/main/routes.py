from app.main import bp


@bp.route('/')
def index():
    return '<h1>This is The Main Blueprint</h1>'