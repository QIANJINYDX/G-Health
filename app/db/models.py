from datetime import datetime
from .db import db

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # 关联聊天会话
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True)
    # 关联用户文件
    files = db.relationship('UserFile', backref='user', lazy=True)
    # 关联风险评估记录
    risk_assessments = db.relationship('RiskAssessment', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

class UserFile(db.Model):
    """用户文件"""
    __tablename__ = 'user_files'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)  # 原始文件名
    stored_filename = db.Column(db.String(255), nullable=False)  # 存储的文件名
    file_path = db.Column(db.String(500), nullable=False)  # 文件存储路径
    file_size = db.Column(db.BigInteger, nullable=False)  # 文件大小（字节）
    file_type = db.Column(db.String(50), nullable=False)  # 文件类型（扩展名）
    mime_type = db.Column(db.String(100), nullable=True)  # MIME类型
    description = db.Column(db.Text, nullable=True)  # 文件描述
    is_public = db.Column(db.Boolean, default=False)  # 是否公开
    download_count = db.Column(db.Integer, default=0)  # 下载次数
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联聊天会话（如果文件是在聊天中上传的）
    chat_session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=True)
    chat_message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=True)

    def __repr__(self):
        return f'<UserFile {self.filename}>'
    
    def get_file_size_display(self):
        """获取文件大小的可读格式"""
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

class ChatSession(db.Model):
    """聊天会话"""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联消息
    messages = db.relationship('ChatMessage', backref='session', lazy=True)
    # 关联文件
    files = db.relationship('UserFile', backref='session', lazy=True)
    # 关联风险评估记录
    risk_assessments = db.relationship('RiskAssessment', backref='session', lazy=True)

    def __repr__(self):
        return f'<ChatSession {self.id}>'

class ChatMessage(db.Model):
    """聊天消息"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)  # True表示用户消息，False表示AI回复
    message_type = db.Column(db.Integer, default=0)  # 0表示正常对话，1表示上传文件，2表示风险评估提示，3表示需要显示评估按钮的消息
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feedback_status = db.Column(db.Integer, default=0)  # 0: 无反馈, 1: 赞同, -1: 反对
    is_visible = db.Column(db.Boolean, default=True)  # True表示显示消息，False表示隐藏消息
    risk_model = db.Column(db.Integer, nullable=True)  # 风险模型ID，仅当message_type=3时有效
    references = db.Column(db.JSON, nullable=True)  # 存储参考文献的JSON数据
    mcp_response = db.Column(db.JSON, nullable=True)  # 存储MCP工具调用响应数据
    has_image = db.Column(db.Boolean, default=False)  # 是否存在图片
    image_data = db.Column(db.Text, nullable=True)  # 图片数据（base64编码）
    follow_up_questions = db.Column(db.JSON, nullable=True)  # 存储AI回复对应的引导问题列表
    # 展示相关
    display_mode = db.Column(db.String(50), default='default')  # 'default' | 'workflow'
    stages = db.Column(db.JSON, nullable=True)  # 工作流阶段列表：[ {id,title,content} ]
    
    # 关联文件
    files = db.relationship('UserFile', backref='message', lazy=True)

    def __repr__(self):
        return f'<ChatMessage {self.id}>'

class RiskAssessment(db.Model):
    """风险评估记录"""
    __tablename__ = 'risk_assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    model_id = db.Column(db.Integer, nullable=False)  # 风险模型ID
    model_name = db.Column(db.String(100), nullable=False)  # 模型名称
    model_name_zh = db.Column(db.String(100), nullable=False)  # 模型中文名称
    
    # 表单数据
    form_data = db.Column(db.JSON, nullable=False)  # 用户填写的表单数据
    
    # 预测结果
    prediction = db.Column(db.String(100), nullable=True)  # 预测结果
    prediction_label = db.Column(db.String(100), nullable=True)  # 预测标签（中文）
    probability = db.Column(db.Float, nullable=True)  # 预测概率
    confidence = db.Column(db.Float, nullable=True)  # 置信度
    
    # 特征重要性分析
    shap_plot = db.Column(db.Text, nullable=True)  # SHAP图表（base64编码）
    
    # 护士建议
    nurse_response = db.Column(db.Text, nullable=True)  # 护士建议内容
    
    # 状态
    status = db.Column(db.String(20), default='completed')  # 状态：pending, completed, failed
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<RiskAssessment {self.id} - {self.model_name_zh}>'
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_name_zh': self.model_name_zh,
            'form_data': self.form_data,
            'prediction': self.prediction,
            'prediction_label': self.prediction_label,
            'probability': self.probability,
            'confidence': self.confidence,
            'shap_plot': self.shap_plot,
            'nurse_response': self.nurse_response,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 