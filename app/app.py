from flask import Flask
from app.config.config import get_config_by_name
from app.initialize_functions import initialize_route, initialize_db, initialize_swagger
import os

def create_app(config=None) -> Flask:
    """
    Create a Flask application.

    Args:
        config: The configuration object to use.

    Returns:
        A Flask application instance.
    """
    # https://nat-notebook-inspire.sii.edu.cn/ws-6040202d-b785-4b37-98b0-c68d65dd52ce/project-d304bac5-9f1c-48cf-b5c1-615d81b66f27/user-5353e278-5dfb-4b1d-827d-add0f39847ce/vscode/0042b4c3-633f-4f89-ba9c-1a9ba82549fd/adda3d65-5638-4bb0-bd9e-4208b681d47f/proxy/5000/
    
    app = Flask(__name__)
    if config:
        app.config.from_object(get_config_by_name(config))

    # 设置session密钥
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')  # 使用环境变量或固定值

    # Initialize extensions
    initialize_db(app)

    # Register blueprints
    initialize_route(app)

    # Initialize Swagger
    initialize_swagger(app)

    # 配置文件上传 - 确保在配置加载后设置
    if 'UPLOAD_FOLDER' not in app.config:
        app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    if 'MAX_CONTENT_LENGTH' not in app.config:
        app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # 配置风险评估模型服务地址
    app.config['RISK_MODEL_SERVICE_URL'] = "http://localhost:5002"

    return app
