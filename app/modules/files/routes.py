import os
from flask import Blueprint, request, jsonify, send_file, current_app, session
from werkzeug.exceptions import NotFound, Forbidden
from app.modules.files.models import FileService
from app.modules.auth.route import login_required

# 创建蓝图
files_bp = Blueprint('files', __name__, url_prefix='/api/v1/files')

@files_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """上传文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 获取可选参数
    session_id = request.form.get('session_id', type=int)
    message_id = request.form.get('message_id', type=int)
    description = request.form.get('description', '')
    
    # 保存文件
    user_file, error = FileService.save_file(
        file=file,
        user_id=session['user_id'],
        session_id=session_id,
        message_id=message_id,
        description=description
    )
    
    if error:
        return jsonify({'error': error}), 400
    
    return jsonify({
        'success': True,
        'file': {
            'id': user_file.id,
            'filename': user_file.filename,
            'file_size': user_file.get_file_size_display(),
            'file_type': user_file.file_type,
            'created_at': user_file.created_at.isoformat(),
            'description': user_file.description
        }
    }), 201

@files_bp.route('/list', methods=['GET'])
@login_required
def get_files():
    """获取用户文件列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    file_type = request.args.get('file_type')
    
    files_pagination = FileService.get_user_files(
        user_id=session['user_id'],
        page=page,
        per_page=per_page,
        file_type=file_type
    )
    
    files = []
    for user_file in files_pagination.items:
        files.append({
            'id': user_file.id,
            'filename': user_file.filename,
            'file_size': user_file.get_file_size_display(),
            'file_type': user_file.file_type,
            'mime_type': user_file.mime_type,
            'description': user_file.description,
            'download_count': user_file.download_count,
            'created_at': user_file.created_at.isoformat(),
            'updated_at': user_file.updated_at.isoformat()
        })
    
    return jsonify({
        'files': files,
        'pagination': {
            'page': files_pagination.page,
            'pages': files_pagination.pages,
            'per_page': files_pagination.per_page,
            'total': files_pagination.total,
            'has_next': files_pagination.has_next,
            'has_prev': files_pagination.has_prev
        }
    })

@files_bp.route('/<int:file_id>', methods=['GET'])
@login_required
def get_file(file_id):
    """获取文件详情"""
    user_file = FileService.get_file_by_id(file_id, session['user_id'])
    if not user_file:
        return jsonify({'error': '文件不存在或无权限'}), 404
    
    return jsonify({
        'id': user_file.id,
        'filename': user_file.filename,
        'file_size': user_file.get_file_size_display(),
        'file_type': user_file.file_type,
        'mime_type': user_file.mime_type,
        'description': user_file.description,
        'download_count': user_file.download_count,
        'is_public': user_file.is_public,
        'created_at': user_file.created_at.isoformat(),
        'updated_at': user_file.updated_at.isoformat()
    })

@files_bp.route('/<int:file_id>/download', methods=['GET'])
@login_required
def download_file(file_id):
    """下载文件"""
    user_file = FileService.get_file_by_id(file_id, session['user_id'])
    if not user_file:
        return jsonify({'error': '文件不存在或无权限'}), 404
    
    if not os.path.exists(user_file.file_path):
        return jsonify({'error': '文件不存在'}), 404
    
    # 增加下载次数
    FileService.increment_download_count(file_id)
    
    return send_file(
        user_file.file_path,
        as_attachment=True,
        download_name=user_file.filename,
        mimetype=user_file.mime_type
    )

@files_bp.route('/<int:file_id>', methods=['PUT'])
@login_required
def update_file(file_id):
    """更新文件信息"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '无效的请求数据'}), 400
    
    # 只允许更新特定字段
    allowed_fields = ['description', 'is_public']
    update_data = {k: v for k, v in data.items() if k in allowed_fields}
    
    success, error = FileService.update_file_info(file_id, session['user_id'], **update_data)
    
    if not success:
        return jsonify({'error': error}), 400
    
    return jsonify({'success': True, 'message': '文件信息更新成功'})

@files_bp.route('/<int:file_id>', methods=['DELETE'])
@login_required
def delete_file(file_id):
    """删除文件"""
    success, error = FileService.delete_file(file_id, session['user_id'])
    
    if not success:
        return jsonify({'error': error}), 400
    
    return jsonify({'success': True, 'message': '文件删除成功'})

@files_bp.route('/stats', methods=['GET'])
@login_required
def get_file_stats():
    """获取用户文件统计信息"""
    stats = FileService.get_file_stats(session['user_id'])
    
    return jsonify({
        'total_files': stats['total_files'],
        'total_size': stats['total_size'],
        'total_size_display': FileService.format_size(stats['total_size']),
        'type_stats': stats['type_stats']
    })

@files_bp.route('/types', methods=['GET'])
def get_supported_types():
    """获取支持的文件类型"""
    return jsonify({
        'supported_types': list(FileService.ALLOWED_EXTENSIONS)
    })

@files_bp.route('/message/<int:message_id>', methods=['GET'])
@login_required
def get_message_files(message_id):
    """获取与特定消息关联的文件列表"""
    try:
        # 获取与消息关联的文件
        from app.db.models import UserFile
        files = UserFile.query.filter_by(
            chat_message_id=message_id,
            user_id=session['user_id']
        ).all()
        
        file_list = []
        for user_file in files:
            file_list.append({
                'id': user_file.id,
                'filename': user_file.filename,
                'file_size': user_file.get_file_size_display(),
                'file_type': user_file.file_type,
                'mime_type': user_file.mime_type,
                'description': user_file.description,
                'download_count': user_file.download_count,
                'created_at': user_file.created_at.isoformat(),
                'updated_at': user_file.updated_at.isoformat()
            })
        
        return jsonify({
            'files': file_list,
            'total': len(file_list)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取消息文件失败: {str(e)}'}), 500

@files_bp.route('/session/<int:session_id>', methods=['GET'])
@login_required
def get_session_files(session_id):
    """获取与特定会话关联的文件列表"""
    try:
        # 获取与会话关联的文件
        from app.db.models import UserFile
        files = UserFile.query.filter_by(
            chat_session_id=session_id,
            user_id=session['user_id']
        ).all()
        
        file_list = []
        for user_file in files:
            file_list.append({
                'id': user_file.id,
                'filename': user_file.filename,
                'file_size': user_file.get_file_size_display(),
                'file_type': user_file.file_type,
                'mime_type': user_file.mime_type,
                'description': user_file.description,
                'download_count': user_file.download_count,
                'created_at': user_file.created_at.isoformat(),
                'updated_at': user_file.updated_at.isoformat()
            })
        
        return jsonify({
            'files': file_list,
            'total': len(file_list)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取会话文件失败: {str(e)}'}), 500

# 添加格式化文件大小的静态方法
@staticmethod
def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

# 将格式化方法添加到FileService类
FileService.format_size = format_size 