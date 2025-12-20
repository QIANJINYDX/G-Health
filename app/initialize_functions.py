from flask import Flask, render_template, redirect, url_for, session,Blueprint
from flasgger import Swagger
from app.modules.main.route import main_bp
from app.modules.chat import chat_bp
from app.modules.auth import auth_bp
from app.modules.files.routes import files_bp
from app.db.db import init_db


def initialize_route(app: Flask):
    with app.app_context():
        @app.route('/')
        def index():
            if 'user_id' in session:
                return redirect(url_for('chat.chat'))
            return redirect(url_for('auth.login_page'))
            
        app.register_blueprint(main_bp, url_prefix='/api/v1/main')
        app.register_blueprint(chat_bp, url_prefix='/api/v1/chat')
        app.register_blueprint(auth_bp, url_prefix='/api/v1/auth')
        app.register_blueprint(files_bp, url_prefix='/api/v1/files')



def initialize_db(app: Flask):
    init_db(app)


def initialize_swagger(app: Flask):
    with app.app_context():
        swagger = Swagger(app)
        return swagger