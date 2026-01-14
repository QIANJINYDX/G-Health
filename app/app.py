from flask import Flask, request
from app.config.config import get_config_by_name
from app.initialize_functions import initialize_route, initialize_db, initialize_swagger
import os
from werkzeug.middleware.proxy_fix import ProxyFix


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
    app.secret_key = os.environ.get(
        'FLASK_SECRET_KEY', 'your-secret-key-here')  # 使用环境变量或固定值

    # 配置 Flask 正确处理反向代理（用于生成正确的URL）
    # 如果设置了 SERVER_NAME，Flask 会使用它来生成绝对URL
    # 否则会从请求头中获取
    app.config['PREFERRED_URL_SCHEME'] = os.environ.get(
        'PREFERRED_URL_SCHEME', 'https')

    # 如果设置了 SERVER_NAME，使用它（生产环境建议设置）
    if os.environ.get('SERVER_NAME'):
        app.config['SERVER_NAME'] = os.environ.get('SERVER_NAME')

    # 信任反向代理转发的协议/Host 等信息（用于生成正确的绝对 URL、scheme）
    # 注意：开启后会信任客户端可伪造的 X-Forwarded-* 头，因此只应在真实反向代理后使用。
    # 默认是否启用 ProxyFix：以 create_app(config) 传入的配置名为准，而不是依赖 FLASK_ENV。
    # 线上通过 run.py 传入 config='production'，但不一定会设置 FLASK_ENV，导致 ProxyFix 误关闭。
    config_name = str(config).lower() if config else ''
    default_proxy_fix = '1' if config_name == 'production' else '0'
    enable_proxy_fix = os.environ.get('ENABLE_PROXY_FIX', default_proxy_fix).lower() in (
        '1', 'true', 'yes', 'y'
    )
    if enable_proxy_fix:
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=1,
            x_proto=1,
            x_host=1,
            x_port=1,
            x_prefix=1,
        )

    # 仅在显式开启时打印代理相关头，方便排查线上 Host/Scheme 异常
    if os.environ.get('DEBUG_PROXY_HEADERS', '0').lower() in ('1', 'true', 'yes', 'y'):
        @app.before_request
        def _debug_proxy_headers():
            # 注意：不要打印 Cookie/Authorization
            current = {
                'url': request.url,
                'host': request.host,
                'scheme': request.scheme,
                'remote_addr': request.remote_addr,
                'xf_host': request.headers.get('X-Forwarded-Host'),
                'xf_proto': request.headers.get('X-Forwarded-Proto'),
                'xf_port': request.headers.get('X-Forwarded-Port'),
                'xf_for': request.headers.get('X-Forwarded-For'),
                'xf_prefix': request.headers.get('X-Forwarded-Prefix'),
            }
            print('[proxy-debug]', current)

    # Initialize extensions
    initialize_db(app)

    # Register blueprints
    initialize_route(app)

    # Initialize Swagger
    initialize_swagger(app)

    # 配置文件上传 - 确保在配置加载后设置
    if 'UPLOAD_FOLDER' not in app.config:
        app.config['UPLOAD_FOLDER'] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'uploads')
    if 'MAX_CONTENT_LENGTH' not in app.config:
        app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # 配置风险评估模型服务地址
    app.config['RISK_MODEL_SERVICE_URL'] = "http://localhost:5002"

    return app
