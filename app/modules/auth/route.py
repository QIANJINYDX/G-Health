from flask import render_template, request, jsonify, session, redirect, url_for
from . import auth_bp
from .controller import AuthController
from functools import wraps

auth_controller = AuthController()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'message': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET'])
def login_page():
    """登录页面路由"""
    if 'user_id' in session:
        # 登录后跳转到聊天页：使用相对路径，避免反向代理未正确传递 Host
        # 导致 _external=True 生成 http://127.0.0.1... 这样的错误绝对地址。
        # 统一带尾斜杠，避免 Flask 自动补斜杠重定向时生成错误的绝对 Location（如 http://127.0.0.1/...）
        return redirect('/api/v1/chat/')
    return render_template('login.html')

@auth_bp.route('/login', methods=['POST'])
def login():
    """登录接口
    ---
    tags:
      - Auth
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            username:
              type: string
              description: 用户名
            password:
              type: string
              description: 密码
    responses:
      200:
        description: 登录成功
      401:
        description: 登录失败
    """
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    result = auth_controller.login(username, password)
    if result.get('success'):
        session['user_id'] = result.get('user_id')
        return jsonify({'message': '登录成功'})
    return jsonify({'message': result.get('message')}), 401

@auth_bp.route('/register', methods=['POST'])
def register():
    """注册接口
    ---
    tags:
      - Auth
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            username:
              type: string
              description: 用户名
            password:
              type: string
              description: 密码
    responses:
      200:
        description: 注册成功
      400:
        description: 注册失败
    """
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    result = auth_controller.register(username, password)
    if result.get('success'):
        return jsonify({'message': '注册成功'})
    return jsonify({'message': result.get('message')}), 400

@auth_bp.route('/logout')
def logout():
    """退出登录
    ---
    tags:
      - Auth
    responses:
      200:
        description: 退出成功
    """
    session.pop('user_id', None)
    # 使用相对路径，避免反向代理未正确传递 Host 导致生成错误的绝对地址
    return redirect('/api/v1/auth/login')

@auth_bp.route('/user-info', methods=['GET'])
@login_required
def get_user_info():
    """获取用户信息
    ---
    tags:
      - Auth
    responses:
      200:
        description: 成功获取用户信息
      401:
        description: 未登录
    """
    result = auth_controller.get_user_info(session['user_id'])
    if result.get('success'):
        return jsonify(result.get('user'))
    return jsonify({'message': result.get('message')}), 404

@auth_bp.route('/settings', methods=['GET'])
@login_required
def settings_page():
    """设置页面路由"""
    return render_template('settings.html')

@auth_bp.route('/profile', methods=['GET'])
@login_required
def profile_page():
    """个人中心页面路由"""
    return render_template('profile.html')

@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """修改密码
    ---
    tags:
      - Auth
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            current_password:
              type: string
              description: 当前密码
            new_password:
              type: string
              description: 新密码
    responses:
      200:
        description: 修改成功
      400:
        description: 修改失败
    """
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not current_password or not new_password:
        return jsonify({'message': '当前密码和新密码不能为空'}), 400
        
    result = auth_controller.change_password_with_verification(session['user_id'], current_password, new_password)
    if result.get('success'):
        return jsonify({'message': '密码修改成功'})
    return jsonify({'message': result.get('message')}), 400

@auth_bp.route('/user-stats', methods=['GET'])
@login_required
def get_user_stats():
    """获取用户使用统计
    ---
    tags:
      - Auth
    responses:
      200:
        description: 成功获取用户统计
      401:
        description: 未登录
    """
    result = auth_controller.get_user_stats(session['user_id'])
    if result.get('success'):
        return jsonify(result.get('stats'))
    return jsonify({'message': result.get('message')}), 404

@auth_bp.route('/chat-stats-by-date', methods=['GET'])
@login_required
def get_chat_stats_by_date():
    """获取按日期分组的对话统计
    ---
    tags:
      - Auth
    parameters:
      - name: days
        in: query
        type: integer
        description: 统计最近多少天的数据，默认30天
    responses:
      200:
        description: 成功获取日期统计
      401:
        description: 未登录
    """
    days = request.args.get('days', 30, type=int)
    result = auth_controller.get_chat_stats_by_date(session['user_id'], days)
    if result.get('success'):
        return jsonify(result.get('data'))
    return jsonify({'message': result.get('message')}), 404

@auth_bp.route('/message-stats-by-date', methods=['GET'])
@login_required
def get_message_stats_by_date():
    """获取按日期分组的消息统计
    ---
    tags:
      - Auth
    parameters:
      - name: days
        in: query
        type: integer
        description: 统计最近多少天的数据，默认30天
    responses:
      200:
        description: 成功获取消息日期统计
      401:
        description: 未登录
    """
    days = request.args.get('days', 30, type=int)
    result = auth_controller.get_message_stats_by_date(session['user_id'], days)
    if result.get('success'):
        return jsonify(result.get('data'))
    return jsonify({'message': result.get('message')}), 404

@auth_bp.route('/risk-assessment-stats-by-date', methods=['GET'])
@login_required
def get_risk_assessment_stats_by_date():
    """获取按日期分组的风险评估统计
    ---
    tags:
      - Auth
    parameters:
      - name: days
        in: query
        type: integer
        description: 统计最近多少天的数据，默认30天
    responses:
      200:
        description: 成功获取风险评估日期统计
      401:
        description: 未登录
    """
    days = request.args.get('days', 30, type=int)
    result = auth_controller.get_risk_assessment_stats_by_date(session['user_id'], days)
    if result.get('success'):
        return jsonify(result.get('data'))
    return jsonify({'message': result.get('message')}), 404 