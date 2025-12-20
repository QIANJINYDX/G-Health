import os
import uuid
import mimetypes
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app, request
from app.db.models import UserFile, User, ChatSession, ChatMessage
from app.db.db import db

class FileService:
    """文件服务类"""
    
    ALLOWED_EXTENSIONS = {
        'pdf', 'doc', 'docx', 'txt', 'csv', 'xls', 'xlsx', 
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff',
        'mp3', 'mp4', 'avi', 'mov', 'wav',
        'zip', 'rar', '7z'
    }
    
    @staticmethod
    def allowed_file(filename):
        """检查文件类型是否允许"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in FileService.ALLOWED_EXTENSIONS
    
    @staticmethod
    def get_file_extension(filename):
        """获取文件扩展名"""
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    @staticmethod
    def generate_stored_filename(original_filename):
        """生成存储文件名"""
        ext = FileService.get_file_extension(original_filename)
        unique_id = str(uuid.uuid4())
        return f"{unique_id}.{ext}" if ext else unique_id
    
    @staticmethod
    def get_upload_folder():
        """获取上传文件夹路径"""
        upload_folder = os.path.join(current_app.root_path, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        return upload_folder
    
    @staticmethod
    def save_file(file, user_id, session_id=None, message_id=None, description=None):
        """保存文件到数据库和文件系统"""
        if not file or not FileService.allowed_file(file.filename):
            return None, "不支持的文件类型"
        
        try:
            # 生成安全的文件名
            original_filename = secure_filename(file.filename)
            stored_filename = FileService.generate_stored_filename(original_filename)
            
            # 确保上传目录存在
            upload_folder = FileService.get_upload_folder()
            file_path = os.path.join(upload_folder, stored_filename)
            
            # 保存文件到文件系统
            file.save(file_path)
            
            # 获取文件信息
            file_size = os.path.getsize(file_path)
            file_type = FileService.get_file_extension(original_filename)
            mime_type = mimetypes.guess_type(original_filename)[0]
            
            # 创建数据库记录
            user_file = UserFile(
                user_id=user_id,
                filename=original_filename,
                stored_filename=stored_filename,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                mime_type=mime_type,
                description=description,
                chat_session_id=session_id,
                chat_message_id=message_id
            )
            
            db.session.add(user_file)
            db.session.commit()
            
            return user_file, None
            
        except Exception as e:
            db.session.rollback()
            return None, f"保存文件失败: {str(e)}"
    
    @staticmethod
    def get_user_files(user_id, page=1, per_page=20, file_type=None):
        """获取用户的文件列表"""
        query = UserFile.query.filter_by(user_id=user_id)
        
        if file_type:
            query = query.filter_by(file_type=file_type)
        
        return query.order_by(UserFile.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
    
    @staticmethod
    def get_file_by_id(file_id, user_id=None):
        """根据ID获取文件"""
        query = UserFile.query.filter_by(id=file_id)
        if user_id:
            query = query.filter_by(user_id=user_id)
        return query.first()
    
    @staticmethod
    def delete_file(file_id, user_id):
        """删除文件"""
        user_file = FileService.get_file_by_id(file_id, user_id)
        if not user_file:
            return False, "文件不存在或无权限"
        
        try:
            # 删除文件系统中的文件
            if os.path.exists(user_file.file_path):
                os.remove(user_file.file_path)
            
            # 删除数据库记录
            db.session.delete(user_file)
            db.session.commit()
            
            return True, None
            
        except Exception as e:
            db.session.rollback()
            return False, f"删除文件失败: {str(e)}"
    
    @staticmethod
    def update_file_info(file_id, user_id, **kwargs):
        """更新文件信息"""
        user_file = FileService.get_file_by_id(file_id, user_id)
        if not user_file:
            return False, "文件不存在或无权限"
        
        try:
            for key, value in kwargs.items():
                if hasattr(user_file, key):
                    setattr(user_file, key, value)
            
            user_file.updated_at = datetime.utcnow()
            db.session.commit()
            
            return True, None
            
        except Exception as e:
            db.session.rollback()
            return False, f"更新文件信息失败: {str(e)}"
    
    @staticmethod
    def increment_download_count(file_id):
        """增加下载次数"""
        user_file = UserFile.query.get(file_id)
        if user_file:
            user_file.download_count += 1
            db.session.commit()
    
    @staticmethod
    def get_file_stats(user_id):
        """获取用户文件统计信息"""
        total_files = UserFile.query.filter_by(user_id=user_id).count()
        total_size = db.session.query(db.func.sum(UserFile.file_size)).filter_by(user_id=user_id).scalar() or 0
        
        # 按文件类型统计
        type_stats = db.session.query(
            UserFile.file_type,
            db.func.count(UserFile.id).label('count')
        ).filter_by(user_id=user_id).group_by(UserFile.file_type).all()
        
        return {
            'total_files': total_files,
            'total_size': total_size,
            'type_stats': dict(type_stats)
        } 