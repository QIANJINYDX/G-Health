import os

from app.app import create_app


# Gunicorn entrypoint: "wsgi:app"
# 支持通过环境变量选择配置：
# - FLASK_CONFIG: development / production / testing
# - 或兼容 FLASK_ENV: development / production
config_name = os.environ.get("FLASK_CONFIG") or os.environ.get("FLASK_ENV") or "production"

app = create_app(config_name)

