from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    db.init_app(app)
    
    # 导入模型以确保它们被注册
    from app.db.models import User, ChatSession, ChatMessage, UserFile, RiskAssessment
    
    with app.app_context():
        # 创建所有表
        db.create_all()
