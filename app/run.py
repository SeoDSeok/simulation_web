from flask import Flask
from app.routes.main import main
from app.routes.scenario import scenario
import os


def create_app():
    app = Flask(__name__)

    # secret_key
    app.secret_key = 'f2f1d23a7c6b4c29f48c8d12e1c75e7f'

    # Blueprint 등록
    app.register_blueprint(main)
    app.register_blueprint(scenario)

    # 정적/템플릿 경로 설정
    app.static_folder = 'app/static'
    app.template_folder = 'app/templates'

    return app


if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    app = create_app()
    app.run(host='0.0.0.0', port=9091, debug=True)