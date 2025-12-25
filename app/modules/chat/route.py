from flask import render_template, request, jsonify, session, redirect, url_for, current_app, send_file, Response, stream_template
from . import chat_bp
from .controller import ChatController
from app.modules.auth.controller import AuthController
from app.db.db import db  # 这里导入的是 SQLAlchemy 实例
from app.db.models import ChatSession, RiskAssessment
from functools import wraps
import os
from werkzeug.utils import secure_filename
from openai import OpenAI
from app.util.file_detection import detect_pdf_content, detect_office_content, detect_image_content
from app.util.clinical_analyst import analyze_dialogue, get_nurse_response, generate_health_report, chat_with_llm, is_call_report_workflow, report_workflow_stream
from app.config.risk_assessment.config import risk_config
from app.util.agent_config import NURSE_PROMPT_IMAGE, get_prompt
import requests
import shap
from ollama import AsyncClient
from ollama import Client
import re
from app.config.risk_assessment.config import risk_types
from PIL import Image
import base64
import io
import json
import time

def predict_image_type_via_api(image_path, risk_model_service_url, threshold=0.9):
    """
    通过API调用风险评估服务预测图片类型
    
    Args:
        image_path: 图片路径
        risk_model_service_url: 风险评估服务URL
        threshold: 置信度阈值
    
    Returns:
        图片类型字符串
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'threshold': str(threshold)}
            response = requests.post(
                f"{risk_model_service_url}/predict-image-type",
                files=files,
                data=data,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('image_type', 'Other_PICTURE')
            else:
                print(f"预测图片类型API调用失败: {response.status_code}")
                return "Other_PICTURE"
    except Exception as e:
        print(f"预测图片类型时出错: {str(e)}")
        return "Other_PICTURE"

def predict_image_via_api(image_path, img_type, risk_model_service_url):
    """
    通过API调用风险评估服务进行图片预测
    
    Args:
        image_path: 图片路径
        img_type: 图片类型
        risk_model_service_url: 风险评估服务URL
    
    Returns:
        预测结果字典
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'image_type': img_type}
            response = requests.post(
                f"{risk_model_service_url}/predict-image",
                files=files,
                data=data,
                timeout=300
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"图片预测API调用失败: {response.status_code}")
                return {"info": "预测失败", "label": "Error"}
    except Exception as e:
        print(f"图片预测时出错: {str(e)}")
        return {"info": f"预测出错: {str(e)}", "label": "Error"}

def pil_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


chat_controller = ChatController()
auth_controller = AuthController()

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    timeout=30,  # 设置超时时间为30秒
    max_retries=3  # 添加重试机制
)

ollama_client = Client(
    host="http://localhost:11434",
    timeout=300
)

# # 初始化OpenAI客户端
# client = OpenAI(
#     api_key="0",
#     base_url=f"http://localhost:{os.environ.get('API_PORT', '8000')}/v1",
#     timeout=30,  # 设置超时时间为30秒
#     max_retries=3  # 添加重试机制
# )

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login_page'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """检查文件类型是否允许上传"""
    ALLOWED_EXTENSIONS = {
        # 文档文件
        'pdf',  # PDF文件
        'doc', 'docx',  # Word文件
        'txt',  # 文本文件
        # 图片文件
        'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp',
        # 其他常见文档
        'xls', 'xlsx',  # Excel文件
        'ppt', 'pptx'   # PowerPoint文件
    }
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@chat_bp.route('/')
def chat():
    """聊天页面"""
    # 允许未登录用户访问，但会在前端检查登录状态
    return render_template('chat.html')

@chat_bp.route('/check-auth', methods=['GET'])
def check_auth():
    """检查用户登录状态"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user_id': session['user_id']
        })
    else:
        return jsonify({
            'authenticated': False
        })

@chat_bp.route('/sessions', methods=['GET'])
@login_required
def get_sessions():
    """获取用户的所有聊天会话"""
    sessions = chat_controller.get_user_sessions(session['user_id'])
    return jsonify([{
        'id': s.id,
        'title': s.title,
        'created_at': s.created_at.isoformat(),
        'updated_at': s.updated_at.isoformat()
    } for s in sessions])

@chat_bp.route('/sessions', methods=['POST'])
@login_required
def create_session():
    """创建新的聊天会话"""
    chat_session = chat_controller.create_session(session['user_id'])
    return jsonify({
        'id': chat_session.id,
        'title': chat_session.title,
        'created_at': chat_session.created_at.isoformat()
    })

@chat_bp.route('/sessions/<int:session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    """删除聊天会话"""
    if chat_controller.delete_session(session_id):
        return jsonify({'message': '删除成功'})
    return jsonify({'message': '会话不存在'}), 404

@chat_bp.route('/sessions/<int:session_id>/messages', methods=['GET'])
@login_required
def get_messages(session_id):
    """获取会话的所有消息"""
    messages = chat_controller.get_session_messages(session_id)
    
    message_list = []
    for m in messages:
        message_data = {
            'id': m.id,
            'content': m.content,
            'role': 'user' if m.is_user else 'assistant',
            'message_type': m.message_type,
            'feedback_status': m.feedback_status,
            'created_at': m.created_at.isoformat(),
            'risk_model': m.risk_model if m.message_type == 3 else None,  # 使用消息中存储的风险模型ID
            'references': m.references,  # 添加参考文献
            'mcp_response': getattr(m, 'mcp_response', None),  # MCP工具调用响应
            'has_image': m.has_image,  # 是否有图片
            'image_data': m.image_data,  # 图片数据
            'follow_up_questions': m.follow_up_questions,  # 引导问题
            'display_mode': getattr(m, 'display_mode', 'default'),
            'stages': getattr(m, 'stages', None)
        }
        
        # 获取与消息关联的文件
        if m.message_type == 1:  # 文件上传消息
            try:
                from app.db.models import UserFile
                files = UserFile.query.filter_by(
                    chat_message_id=m.id,
                    user_id=session['user_id']
                ).all()
                
                message_data['files'] = [{
                    'id': f.id,
                    'filename': f.filename,
                    'file_size': f.get_file_size_display(),
                    'file_type': f.file_type,
                    'mime_type': f.mime_type,
                    'description': f.description
                } for f in files]
            except Exception as e:
                print(f"获取消息文件失败: {str(e)}")
                message_data['files'] = []
        else:
            message_data['files'] = []
        
        message_list.append(message_data)
    
    return jsonify(message_list)



@chat_bp.route('/sessions/<int:session_id>/messages', methods=['POST'])
@login_required
def send_message(session_id):
    """发送消息并返回SSE流式响应"""
    # 在函数开始时获取配置值和请求数据，避免在生成器中使用current_app和request
    upload_folder_base = current_app.config['UPLOAD_FOLDER']
    risk_model_service_url = current_app.config['RISK_MODEL_SERVICE_URL']
    
    # 获取app实例，用于后续的app context
    app = current_app._get_current_object()
    
    # 获取用户ID，避免在生成器中访问session
    user_id = session['user_id']
    
    # 获取请求数据
    message = request.form.get('message', '')
    files = request.files.getlist('files')
    rag_enabled = request.form.get('rag_enabled', '0') == '1'
    deep_think = request.form.get('deep_think', '0') == '1'
    language = request.form.get('language', 'zh')  # 获取语言参数，默认为中文
    model = request.form.get('model', 'qwen3:32b')  # 获取模型参数，默认为 qwen3:32b
    file_paths = []
    saved_files = []  # 存储保存到数据库的文件信息
    
    # 保存文件到数据库和文件系统
    if files:
        from app.modules.files.models import FileService
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                # 使用FileService保存文件到数据库
                user_file, error = FileService.save_file(
                    file=file,
                    user_id=user_id,
                    session_id=session_id,
                    description=f"聊天中上传的文件: {file.filename}"
                )
                
                if user_file:
                    saved_files.append({
                        'id': user_file.id,
                        'filename': user_file.filename,
                        'file_path': user_file.file_path,
                        'file_size': user_file.get_file_size_display(),
                        'file_type': user_file.file_type
                    })
                    file_paths.append(user_file.file_path)
                else:
                    print(f"文件保存失败: {error}")
                    # 如果保存失败，仍然保存到临时目录用于处理
                    temp_path = os.path.join(upload_folder_base, str(session_id), file.filename)
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    file.save(temp_path)
                    file_paths.append(temp_path)
                
    def generate():
        # 已移除 app_context 包裹
        try:
            # 使用在外部函数中已获取的 language 变量（通过闭包访问）
            # language 已在 send_message 函数中从请求中获取
            # 初始化消息变量
            current_message = message  # 用于判断逻辑和消息历史（不包含护士提示词）
            current_message_for_ai = message  # 用于传递给AI服务（包含护士提示词）
            
            if not current_message and not files:
                yield f"data: {json.dumps({'type': 'error', 'message': '消息和文件不能同时为空'})}\n\n"
                return
            
            # 发送开始信号
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            
            # 处理文件上传
            uploaded_files = []
            image_messages = []
            has_risk_assessment_image = False  # 标记是否进行了风险评估图片预测
            risk_assessment_results = []  # 存储风险评估结果（用于后续生成护士建议）
            nurse_response_generated = False  # 标记是否已经生成了护士建议（用于避免重复回复）
            if files:
                show_message = current_message
                is_image_class = False
                upload_folder = os.path.join(upload_folder_base, str(session_id))
                os.makedirs(upload_folder, exist_ok=True)
                ob_image_base64s = []
                
                # 发送文件处理开始信号
                yield f"data: {json.dumps({'type': 'file_processing_start'})}\n\n"
                
                for i, file_path in enumerate(file_paths):
                    filename = secure_filename(os.path.basename(file_path))
                    if file_path and allowed_file(os.path.basename(file_path)):
                        # 使用保存的文件信息
                        if i < len(saved_files):
                            file_info = saved_files[i]
                            uploaded_files.append({
                                'id': file_info['id'],
                                'name': file_info['filename'],
                                'type': file_info['file_type'],
                                'size': file_info['file_size']
                            })
                        else:
                            uploaded_files.append({
                                'name': filename,
                                'type': filename.rsplit('.', 1)[1].lower(),
                                'size': os.path.getsize(file_path)
                            })
                        
                        # 发送文件处理进度
                        yield f"data: {json.dumps({'type': 'file_processing', 'filename': filename})}\n\n"
                        
                        try:
                            file_content = ""  # 用于判断逻辑和消息历史（不包含护士提示词）
                            file_content_for_ai = ""  # 用于传递给AI的内容（包含护士提示词）
                            if filename.lower().endswith('.pdf'):
                                file_content = detect_pdf_content(file_path)
                                file_content_for_ai = file_content
                            elif filename.lower().endswith(('.ppt', '.pptx', '.doc', '.docx')):
                                file_content = detect_office_content(file_path)
                                file_content_for_ai = file_content
                            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                # 通过API调用风险评估服务预测图片类型
                                try:
                                    risk_model_service_url = current_app.config.get('RISK_MODEL_SERVICE_URL', 'http://localhost:5002')
                                    img_type = predict_image_type_via_api(file_path, risk_model_service_url)
                                except Exception as e:
                                    print(f"预测图片类型失败: {str(e)}")
                                    img_type = "Other_PICTURE"
                                
                                if img_type == "Other_PICTURE":
                                    file_content = detect_image_content(file_path)
                                    file_content_for_ai = file_content
                                else:
                                    # 通过API调用风险评估服务进行图片预测
                                    # 标记已进行风险评估图片预测，后续不触发体检报告工作流
                                    has_risk_assessment_image = True
                                    try:
                                        risk_model_service_url = current_app.config.get('RISK_MODEL_SERVICE_URL', 'http://localhost:5002')
                                        file_result = predict_image_via_api(file_path, img_type, risk_model_service_url)
                                        print(f"[风险评估图片预测] 图片类型: {img_type}, 预测结果: {file_result}")
                                        
                                        # 保存风险评估结果，用于后续生成护士建议
                                        risk_assessment_results.append({
                                            'img_type': img_type,
                                            'file_result': file_result,
                                            'filename': filename
                                        })
                                        
                                        # 文件内容不包含护士提示词（用于消息历史和判断逻辑）
                                        file_content = file_result.get("info", "")
                                        # 用于传递给AI的内容包含护士提示词
                                        nurse_prompt_image = get_prompt('NURSE_PROMPT_IMAGE', language)
                                        file_content_for_ai = file_content + nurse_prompt_image
                                        
                                        print(f"[风险评估图片预测] file_content长度: {len(file_content)}, file_content_for_ai长度: {len(file_content_for_ai)}")
                                        
                                        if img_type == "breast_cancer" and "image_base64" in file_result:
                                            is_image_class = True
                                            image_base64 = file_result["image_base64"]
                                            ob_image_base64s.append(image_base64)
                                    except Exception as e:
                                        print(f"图片预测失败: {str(e)}")
                                        file_content = detect_image_content(file_path)
                                        file_content_for_ai = file_content
                            
                            if file_content:
                                # current_message用于消息历史和判断逻辑（不包含护士提示词）
                                current_message = f"{current_message}\n\n以下是上传的文件内容：\n{file_content}" if current_message else f"以下是上传的文件内容：\n{file_content}"
                                # current_message_for_ai用于传递给AI服务（包含护士提示词）
                                current_message_for_ai = f"{current_message_for_ai}\n\n以下是上传的文件内容：\n{file_content_for_ai}" if current_message_for_ai else f"以下是上传的文件内容：\n{file_content_for_ai}"
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'message': f'文件解析错误：{str(e)}'})}\n\n"
                            return
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'不支持的文件类型：{filename}'})}\n\n"
                        return
                
                # 发送文件处理完成信号
                yield f"data: {json.dumps({'type': 'file_processing_complete'})}\n\n"
                
                # 实时推送图片消息到SSE流
                
                if uploaded_files:
                    # 先保存用户消息（包含文件内容和风险评估结果）
                    # 使用更新后的current_message，而不是初始的show_message
                    user_message_content = current_message if current_message else (show_message if show_message else "上传了文件")
                    print(f"[保存用户消息] 消息内容长度: {len(user_message_content)}, 内容预览: {user_message_content[:200]}")
                    user_message = chat_controller.add_message(session_id, user_message_content, is_user=True, message_type=1, is_visible=True)
                    
                    # 将文件关联到用户消息
                    if saved_files:
                        from app.modules.files.models import FileService
                        for file_info in saved_files:
                            # 更新文件记录，关联到当前消息
                            FileService.update_file_info(
                                file_info['id'], 
                                user_id, 
                                chat_message_id=user_message.id
                            )
                    
                    # 然后创建图片消息（如果有），确保它们的创建时间晚于用户消息
                    # 这样在按时间排序时，图片消息会显示在用户消息之后
                    image_messages = []
                    if is_image_class:
                        import time
                        time.sleep(0.01)  # 短暂延迟，确保时间戳不同
                        for image_base64 in ob_image_base64s:
                            image_message = chat_controller.add_message(session_id, "你的检测结果:", is_user=False, message_type=1, is_visible=True, has_image=True, image_data=image_base64)
                            image_messages.append(image_message)

                    if image_messages:
                        for img_msg in image_messages:
                            # 只发送message_id，避免base64数据过大导致JSON截断
                            # 前端会通过API获取完整的图片数据
                            yield f"data: {json.dumps({'type': 'image_message', 'message_id': img_msg.id, 'content': img_msg.content, 'has_image': True})}\n\n"
                    
                    # 对于非breast_cancer类型的风险评估图片，生成护士建议消息
                    if has_risk_assessment_image and risk_assessment_results:
                        print(f"[风险评估图片预测] 开始处理 {len(risk_assessment_results)} 个风险评估结果")
                        # 先发送AI处理开始信号，让前端创建消息容器
                        yield f"data: {json.dumps({'type': 'ai_processing_start'})}\n\n"
                        
                        for risk_result in risk_assessment_results:
                            if risk_result['img_type'] != "breast_cancer":
                                # 生成护士建议
                                try:
                                    file_result = risk_result['file_result']
                                    img_type = risk_result['img_type']
                                    filename = risk_result['filename']
                                    
                                    # 获取预测结果信息
                                    prediction_info = file_result.get("info", "")
                                    prediction_label = file_result.get("label", "")
                                    
                                    print(f"[风险评估图片预测] 处理图片: {filename}, 类型: {img_type}, 预测结果: {prediction_label}, info长度: {len(prediction_info)}")
                                    
                                    if prediction_info or prediction_label:
                                        # 调用get_nurse_response生成护士建议
                                        nurse_response = get_nurse_response(
                                            img_type, 
                                            prediction_label if prediction_label else prediction_info,
                                            str(file_result),
                                            ollama_client,
                                            language,
                                            model=model
                                        )
                                        
                                        if nurse_response:
                                            # 保存护士建议作为消息
                                            import time
                                            time.sleep(0.01)  # 短暂延迟，确保时间戳不同
                                            nurse_message = chat_controller.add_message(
                                                session_id,
                                                nurse_response,
                                                is_user=False,
                                                message_type=0,
                                                is_visible=True
                                            )
                                            # 发送护士建议消息（使用content类型，让前端正常显示）
                                            # 先发送完整内容
                                            yield f"data: {json.dumps({'type': 'content', 'content': nurse_response, 'done': True})}\n\n"
                                            # 然后发送消息完成信号
                                            yield f"data: {json.dumps({'type': 'message_complete', 'message_id': nurse_message.id, 'full_response': nurse_response, 'rag_response': None, 'references': None})}\n\n"
                                            print(f"[风险评估图片预测] 已生成并保存护士建议消息，ID: {nurse_message.id}, 内容长度: {len(nurse_response)}")
                                            nurse_response_generated = True  # 标记已生成护士建议
                                        else:
                                            print(f"[风险评估图片预测] 护士建议为空，跳过保存")
                                    else:
                                        print(f"[风险评估图片预测] 预测结果为空，跳过生成护士建议")
                                except Exception as e:
                                    print(f"[风险评估图片预测] 生成护士建议失败: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                    
            elif current_message != "":
                user_message = chat_controller.add_message(session_id, current_message, is_user=True, message_type=0, is_visible=True)
            
            # 发送AI处理开始信号（如果还没有发送）
            if not nurse_response_generated:
                yield f"data: {json.dumps({'type': 'ai_processing_start'})}\n\n"
            
            # 调用AI服务获取流式回复
            try:
                # 只获取可见消息，不包含文件处理过程中的临时消息
                messages = chat_controller.get_session_messages(session_id, vis=True)
                formatted_messages = []
                last_role = None
                
                
                for msg in messages:
                    current_role = "user" if msg.is_user else "assistant"
                    formatted_messages.append({
                        "role": current_role,
                        "content": msg.content
                    })
                    last_role = current_role
                
                # 如果最后一条消息不是用户消息，添加当前消息（使用包含护士提示词的版本用于AI上下文）
                if formatted_messages and formatted_messages[-1]["role"] != "user":
                    formatted_messages.append({
                        "role": "user",
                        "content": current_message_for_ai  # 使用包含护士提示词的版本
                    })
                # 判断最后一条消息是否涉及体检报告解析、健康指标（如血糖、血压、肝功能、BMI 等）、体检异常解释、健康建议或风险评估
                # 如果已进行风险评估图片预测，则不触发体检报告工作流
                is_call_report = False
                if not has_risk_assessment_image:
                    # 使用不包含护士提示词的版本进行判断
                    print(f"当前消息: {current_message}","体检报告分析工作流触发判断器判断结果:")
                    is_call_report = is_call_report_workflow(current_message, ollama_client, language, model=model)
                    print(f"是否调用体检报告分析工作流: {is_call_report}, 使用模型: {model}")
                else:
                    print(f"已进行风险评估图片预测，跳过体检报告工作流判断")
                
                if is_call_report:
                    # 开始体检报告流式工作流
                    yield f"data: {json.dumps({'type': 'report_workflow_start'})}\n\n"

                    # 构建对话文本（包含历史消息与当前消息）
                    try:
                        dialogue_msgs = chat_controller.get_session_messages(session_id)
                    except Exception:
                        dialogue_msgs = []
                    dialogue_text = ""
                    for m in dialogue_msgs:
                        role_cn = "用户" if m.is_user else "医生"
                        dialogue_text += f"{role_cn}：{m.content}\n"
                    if current_message_for_ai:
                        dialogue_text += f"用户：{current_message_for_ai}\n"

                    # 执行流式工作流并逐段发送
                    workflow_full = ""
                    workflow_stages = []
                    try:
                        stage_index = 0
                        for event in report_workflow_stream(dialogue_text, ollama_client, language, model=model):
                            if not isinstance(event, dict):
                                continue
                            etype = event.get('type')
                            if etype == 'stage':
                                title = event.get('title', 'Stage')
                                content_piece = event.get('content') or ''
                                stage_index += 1
                                workflow_full += f"{title}\n{content_piece}\n\n"
                                workflow_stages.append({
                                    'id': stage_index,
                                    'title': title,
                                    'content': content_piece
                                })
                                yield f"data: {json.dumps({'type': 'stage', 'title': title, 'content': content_piece, 'done': False})}\n\n"
                            elif etype == 'content':
                                content_piece = event.get('content') or ''
                                if content_piece:
                                    workflow_full += content_piece
                                    stage_index += 1
                                    workflow_stages.append({
                                        'id': stage_index,
                                        'title': '内容',
                                        'content': content_piece
                                    })
                                    yield f"data: {json.dumps({'type': 'content', 'content': content_piece, 'done': False})}\n\n"
                    except Exception as wf_error:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'报告工作流错误：{str(wf_error)}'})}\n\n"
                        return

                    # 生成引导问题（可选）
                    follow_up_questions = None
                    try:
                        from app.util.clinical_analyst import generate_follow_up_questions
                        # 使用不包含护士提示词的版本生成后续追问建议
                        follow_up_questions = generate_follow_up_questions(current_message or "", workflow_full, ollama_client, language, model=model)
                    except Exception as follow_up_error:
                        print(f"生成后续追问建议失败: {str(follow_up_error)}")

                    # 保存AI回复
                    ai_message = chat_controller.add_message(
                        session_id,
                        workflow_full,
                        is_user=False,
                        references=None,
                        follow_up_questions=follow_up_questions,
                        display_mode='workflow',
                        stages=workflow_stages
                    )

                    # 通知客户端消息完成
                    yield f"data: {json.dumps({'type': 'message_complete', 'message_id': ai_message.id, 'full_response': workflow_full, 'rag_response': None, 'references': None})}\n\n"

                    # 发送引导问题
                    if follow_up_questions:
                        yield f"data: {json.dumps({'type': 'follow_up_questions', 'questions': follow_up_questions})}\n\n"

                    # 工作流完成与总完成信号
                    yield f"data: {json.dumps({'type': 'report_workflow_complete'})}\n\n"
                    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                    return
                
                # 如果已经生成了护士建议（风险评估图片预测），则跳过后续的AI对话
                if nurse_response_generated:
                    print(f"[风险评估图片预测] 已生成护士建议，跳过后续AI对话")
                    # 发送完成信号
                    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                    return
                
                # 使用流式对话函数
                try:
                    # 发送RAG处理开始信号
                    if rag_enabled:
                        yield f"data: {json.dumps({'type': 'rag_processing_start'})}\n\n"
                    
                    # 打印调用信息
                    print(f"[chat_with_llm] 开始调用AI对话服务")
                    print(f"[chat_with_llm] 模型: {model}")
                    print(f"[chat_with_llm] RAG启用: {rag_enabled}")
                    print(f"[chat_with_llm] 深度思考启用: {deep_think}")
                    print(f"[chat_with_llm] 消息数量: {len(formatted_messages)}")
                    print(f"[chat_with_llm] 流式输出: True")
                    
                    # 调用流式AI对话
                    result = chat_with_llm(
                        messages=formatted_messages,
                        client=ollama_client,
                        model=model,  # 使用从请求中获取的模型参数
                        use_rag=rag_enabled,
                        deep_think=deep_think,
                        stream=True,
                        use_mcp=True  # 启用MCP
                    )
                    
                    # 解析返回值（可能是2个或3个值）
                    try:
                        if isinstance(result, tuple):
                            if len(result) == 3:
                                stream_response, references, mcp_data = result
                            elif len(result) == 2:
                                stream_response, references = result
                                mcp_data = None
                            else:
                                stream_response = result[0] if result else None
                                references = None
                                mcp_data = None
                        else:
                            stream_response = result
                            references = None
                            mcp_data = None
                    except Exception as e:
                        print(f"解析返回值错误: {e}")
                        stream_response = result
                        references = None
                        mcp_data = None
                    
                    # 处理MCP数据，准备用于发送和保存
                    processed_mcp_data = None
                    if mcp_data and (mcp_data.get("tool_logs") or mcp_data.get("result")):
                        print("MCP工具调用日志：", mcp_data.get("tool_logs"))
                        print("MCP结果：", mcp_data.get("result"))
                        
                        # 处理MCP数据，确保安全序列化
                        mcp_result = mcp_data.get('result', '')
                        mcp_tool_logs = mcp_data.get('tool_logs') or []
                        
                        # 确保result是字符串
                        if mcp_result is not None:
                            mcp_result = str(mcp_result)
                        else:
                            mcp_result = ''
                        
                        # 处理tool_logs，确保每个项都是字符串且长度合理
                        processed_tool_logs = []
                        for log in mcp_tool_logs:
                            if log is None:
                                continue
                            log_str = str(log) if not isinstance(log, str) else log
                            # 限制每个日志项的最大长度（5000字符）
                            if len(log_str) > 5000:
                                log_str = log_str[:5000] + "...(已截断)"
                            processed_tool_logs.append(log_str)
                        
                        # 限制result的长度（10000字符）
                        if len(mcp_result) > 10000:
                            mcp_result = mcp_result[:10000] + "...(已截断)"
                        
                        # 保存处理后的数据，用于后续保存到数据库
                        processed_mcp_data = {
                            'result': mcp_result,
                            'tool_logs': processed_tool_logs
                        }
                    
                    # 发送MCP响应到前端
                    if processed_mcp_data:
                        # 发送MCP处理开始信号
                        yield f"data: {json.dumps({'type': 'mcp_processing_start'}, ensure_ascii=False)}\n\n"
                        
                        # 发送MCP工具调用结果
                        try:
                            mcp_response_data = {
                                'type': 'mcp_response',
                                'result': processed_mcp_data['result'],
                                'tool_logs': processed_mcp_data['tool_logs']
                            }
                            # 使用ensure_ascii=False正确处理中文，separators参数减小JSON大小
                            # 使用strict=False允许处理控制字符
                            json_str = json.dumps(mcp_response_data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                            
                            # 验证JSON字符串是否有效
                            json.loads(json_str)  # 如果无效会抛出异常
                            
                            # 确保SSE格式正确：data: {json}\n\n
                            yield f"data: {json_str}\n\n"
                        except (TypeError, ValueError, json.JSONDecodeError) as e:
                            print(f"MCP响应JSON序列化错误: {e}")
                            print(f"原始数据长度 - result: {len(processed_mcp_data.get('result', ''))}, tool_logs: {len(processed_mcp_data.get('tool_logs', []))}")
                            # 如果序列化失败，发送简化版本
                            try:
                                # 进一步简化数据
                                simple_result = processed_mcp_data['result'][:500] if processed_mcp_data.get('result') else ''
                                simple_data = {
                                    'type': 'mcp_response',
                                    'result': simple_result,
                                    'tool_logs': [log[:200] for log in processed_mcp_data.get('tool_logs', [])[:3]]  # 只保留前3个，每个最多200字符
                                }
                                json_str = json.dumps(simple_data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                                yield f"data: {json_str}\n\n"
                            except Exception as e2:
                                print(f"简化版本序列化也失败: {e2}")
                                # 最后的兜底方案：只发送类型
                                yield f"data: {json.dumps({'type': 'mcp_response', 'result': '数据过大，无法显示', 'tool_logs': []}, ensure_ascii=False)}\n\n"
                        
                        # 发送MCP处理完成信号
                        yield f"data: {json.dumps({'type': 'mcp_processing_complete'}, ensure_ascii=False)}\n\n"
                    
                    # 发送RAG处理完成信号
                    if rag_enabled:
                        print("检索到的相关资料：",references)
                        # 立即发送检索到的相关资料
                        yield f"data: {json.dumps({'type': 'rag_references', 'references': references})}\n\n"
                        yield f"data: {json.dumps({'type': 'rag_processing_complete'})}\n\n"
                    
                    # 处理流式响应
                    full_response = ""
                    rag_response = None
                    
                    # 如果是字典格式（非流式），直接处理
                    if isinstance(stream_response, dict):
                        if rag_enabled:
                            rag_response = stream_response.get("rag_response")
                            references = stream_response.get("references")
                            full_response = stream_response.get("llm_response", "")
                            # 立即发送检索到的相关资料
                            yield f"data: {json.dumps({'type': 'rag_references', 'references': references})}\n\n"
                        else:
                            full_response = stream_response
                        
                        # 发送完整响应
                        yield f"data: {json.dumps({'type': 'content', 'content': full_response, 'done': True})}\n\n"
                    else:
                        # 处理真正的流式响应
                        try:
                            think_content_accumulated = ""  # 累积思考内容
                            think_content_sent = False  # 标记是否已发送思考内容
                            
                            for chunk in stream_response:
                                # 先检查是否有思考内容
                                if hasattr(chunk, 'message') and hasattr(chunk.message, 'thinking'):
                                    thinking = chunk.message.thinking
                                    if thinking:
                                        think_content_accumulated += thinking
                                        # 如果思考内容还没有发送过，先发送思考部分
                                        if not think_content_sent:
                                            yield f"data: {json.dumps({'type': 'thinking', 'content': thinking, 'done': False})}\n\n"
                                            think_content_sent = True
                                        else:
                                            # 如果已经发送过，继续追加思考内容
                                            yield f"data: {json.dumps({'type': 'thinking', 'content': thinking, 'done': False})}\n\n"
                                
                                # 然后处理内容部分
                                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                                    content = chunk.message.content
                                    if content:
                                        full_response += content
                                        yield f"data: {json.dumps({'type': 'content', 'content': content, 'done': False})}\n\n"
                        except Exception as stream_error:
                            print(f"Stream processing error: {str(stream_error)}")
                            # 如果流式处理失败，尝试获取完整响应
                            try:
                                if hasattr(stream_response, 'message') and hasattr(stream_response.message, 'content'):
                                    full_response = stream_response.message.content
                                    # 尝试获取思考内容
                                    if hasattr(stream_response, 'message') and hasattr(stream_response.message, 'thinking'):
                                        think_content_accumulated = stream_response.message.thinking or ""
                                    yield f"data: {json.dumps({'type': 'content', 'content': full_response, 'done': True})}\n\n"
                            except:
                                # 如果都失败了，发送错误
                                yield f"data: {json.dumps({'type': 'error', 'message': '流式响应处理失败'})}\n\n"
                                return
                    
                    # 流式传输完成后，将思考内容和回答内容组合
                    # 格式：<think>思考内容</think>回答内容
                    message_content_to_save = full_response
                    if think_content_accumulated:
                        message_content_to_save = f"<think>{think_content_accumulated}</think>{full_response}"
                        print(f"[保存消息] 包含思考内容，思考内容长度: {len(think_content_accumulated)}, 回答内容长度: {len(full_response)}")
                    
                    # 准备MCP响应数据用于保存（使用处理后的数据，进一步限制大小）
                    mcp_response_to_save = None
                    if processed_mcp_data:
                        # 进一步限制保存的数据大小，避免数据库字段过大
                        mcp_response_to_save = {
                            'result': processed_mcp_data['result'],
                            'tool_logs': []
                        }
                        # 限制result长度（5000字符用于保存）
                        if mcp_response_to_save['result'] and len(mcp_response_to_save['result']) > 5000:
                            mcp_response_to_save['result'] = mcp_response_to_save['result'][:5000] + "...(已截断)"
                        # 限制tool_logs数量和长度（最多保存10个，每个最多2000字符）
                        if processed_mcp_data['tool_logs']:
                            for log in processed_mcp_data['tool_logs'][:10]:
                                log_str = str(log) if not isinstance(log, str) else log
                                if len(log_str) > 2000:
                                    log_str = log_str[:2000] + "...(已截断)"
                                mcp_response_to_save['tool_logs'].append(log_str)
                    
                    # 流式传输完成后，生成后续追问建议
                    follow_up_questions = None
                    try:
                        from app.util.clinical_analyst import generate_follow_up_questions
                        
                        # 生成后续追问建议（使用不包含护士提示词的版本）
                        follow_up_questions = generate_follow_up_questions(current_message, full_response, ollama_client, language, model=model)
                        print("流式输出后的引导问题：",follow_up_questions)
                        
                    except Exception as follow_up_error:
                        print(f"生成后续追问建议失败: {str(follow_up_error)}")
                        # 不中断流式传输，只记录错误
                    
                    # 保存AI回复（包含思考内容、引导问题和MCP响应）
                    ai_message = chat_controller.add_message(
                        session_id, 
                        message_content_to_save,  # 使用包含思考内容的完整消息
                        is_user=False,
                        references=references,
                        mcp_response=mcp_response_to_save,
                        follow_up_questions=follow_up_questions
                    )
                    
                    # 发送消息完成信号
                    yield f"data: {json.dumps({'type': 'message_complete', 'message_id': ai_message.id, 'full_response': full_response, 'rag_response': rag_response, 'references': references})}\n\n"
                    
                    # 处理风险评估
                    if current_message:
                        risk_model = analyze_dialogue(current_message, ollama_client, language, model=model)
                        if risk_model >= 0:
                            model_config = risk_config.get_model_info(risk_model, language)
                            if model_config:
                                disease = model_config.get('model_name_display', risk_types.get(risk_model, ''))
                                if language == 'en':
                                    risk_message = (
                                        f"Your description may involve {disease} risk, it is recommended to conduct a relevant assessment.\n"
                                        f"Please click the 'Start Assessment' button below to fill out the assessment form."
                                    )
                                else:
                                    risk_message = (
                                        f"检测到您的描述可能涉及{disease}风险，建议进行相关评估。\n"
                                        f"请点击下方的开始评估按钮，填写评估表单。"
                                        )
                                risk_message_obj = chat_controller.add_message(
                                    session_id,
                                    risk_message,
                                    is_user=False,
                                    message_type=3,
                                    is_visible=True,
                                    risk_model=risk_model
                                )
                                
                                yield f"data: {json.dumps({'type': 'risk_assessment', 'message_id': risk_message_obj.id, 'content': risk_message, 'risk_model': risk_model})}\n\n"
                    
                    # 发送图片消息
                    # if image_messages:
                    #     for img_msg in image_messages:
                    #         yield f"data: {json.dumps({'type': 'image_message', 'message_id': img_msg.id, 'content': img_msg.content, 'has_image': True})}\n\n"
                    
                    # 发送引导问题（如果存在，且还未发送）
                    if follow_up_questions:
                        yield f"data: {json.dumps({'type': 'follow_up_questions', 'questions': follow_up_questions})}\n\n"
                    
                    # 发送最终完成信号
                    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                    
                except Exception as ai_error:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'AI服务错误：{str(ai_error)}'})}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'处理错误：{str(e)}'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'发生错误：{str(e)}'})}\n\n"

    def wrapped_generate():
        with app.app_context():
            yield from generate()

    return Response(wrapped_generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

@chat_bp.route('/models', methods=['GET'])
@login_required
def get_ollama_models():
    """获取可用的Ollama模型列表"""
    try:
        # 获取Ollama模型列表
        response = ollama_client.list()
        
        # 调试：打印响应类型和内容
        print(f"Ollama API响应类型: {type(response)}")
        
        # 安全地提取模型名称和详细信息
        models = []
        
        # Ollama API返回格式可能是：
        # 1. ListResponse对象（有models属性）
        # 2. 字典（包含'models'键）
        # 3. 列表
        model_list = []
        
        # 处理ListResponse对象
        if hasattr(response, 'models'):
            model_list = response.models
            print(f"检测到ListResponse对象，包含 {len(model_list)} 个模型")
        elif isinstance(response, dict):
            if 'models' in response:
                model_list = response['models']
                print(f"检测到字典格式，包含 {len(model_list)} 个模型")
            else:
                print(f"警告：响应字典中没有'models'键，键为: {list(response.keys())}")
        elif isinstance(response, list):
            model_list = response
            print(f"检测到列表格式，包含 {len(model_list)} 个模型")
        else:
            print(f"警告：意外的响应类型: {type(response)}")
        
        # 处理模型列表
        if model_list:
            print(f"开始处理 {len(model_list)} 个模型")
            for model_item in model_list:
                try:
                    # 处理Model对象（有model属性）或字典（有name/model键）
                    if hasattr(model_item, 'model'):
                        # Model对象
                        model_name = model_item.model
                        size = getattr(model_item, 'size', 0)
                        modified_at = str(getattr(model_item, 'modified_at', ''))
                        digest = getattr(model_item, 'digest', '')
                        
                        # 提取details信息（ModelDetails对象）
                        details = getattr(model_item, 'details', None)
                        if details:
                            family = getattr(details, 'family', '')
                            format_val = getattr(details, 'format', '')
                            param_size = getattr(details, 'parameter_size', '')
                            quant_level = getattr(details, 'quantization_level', '')
                        else:
                            family = format_val = param_size = quant_level = ''
                    elif isinstance(model_item, dict):
                        # 字典格式
                        model_name = model_item.get('name') or model_item.get('model')
                        size = model_item.get('size', 0)
                        modified_at = str(model_item.get('modified_at', ''))
                        digest = model_item.get('digest', '')
                        
                        # 提取details信息
                        details = model_item.get('details', {})
                        if isinstance(details, dict):
                            family = details.get('family', '')
                            format_val = details.get('format', '')
                            param_size = details.get('parameter_size', '')
                            quant_level = details.get('quantization_level', '')
                        elif hasattr(details, 'family'):
                            # details是对象
                            family = details.family if hasattr(details, 'family') else ''
                            format_val = details.format if hasattr(details, 'format') else ''
                            param_size = details.parameter_size if hasattr(details, 'parameter_size') else ''
                            quant_level = details.quantization_level if hasattr(details, 'quantization_level') else ''
                        else:
                            family = format_val = param_size = quant_level = ''
                    else:
                        print(f"警告：未知的模型对象类型: {type(model_item)}")
                        continue
                    
                    if model_name:
                        model_info = {
                            'name': model_name,
                            'size': size,
                            'modified_at': modified_at,
                            'digest': digest,
                            'family': family,
                            'format': format_val,
                            'parameter_size': param_size,
                            'quantization_level': quant_level
                        }
                        models.append(model_info)
                        print(f"成功添加模型: {model_name} (大小: {size} bytes)")
                    else:
                        print(f"警告：模型对象缺少名称字段")
                except Exception as e:
                    print(f"处理模型时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        else:
            print(f"错误：无法提取模型列表")
        
        # 按模型名称排序
        models.sort(key=lambda x: x['name'])
        
        print(f"最终返回 {len(models)} 个模型")
        
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"获取Ollama模型列表失败: {error_detail}")
        return jsonify({
            'status': 'error',
            'message': f'获取模型列表失败：{str(e)}',
            'models': [],
            'count': 0,
            'error_detail': str(error_detail)
        }), 500

@chat_bp.route('/test-connection', methods=['GET'])
def test_connection():
    """测试连接"""
    try:
        # 测试Ollama连接
        response = ollama_client.list()
        
        # 安全地提取模型名称
        models = []
        if 'models' in response and isinstance(response['models'], list):
            for model in response['models']:
                if isinstance(model, dict) and 'name' in model:
                    models.append(model['name'])
        
        return jsonify({
            'status': 'success',
            'message': '连接正常',
            'models': models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'连接失败：{str(e)}'
        }), 500

@chat_bp.route('/test-sse', methods=['GET'])
def test_sse():
    """测试SSE连接"""
    def generate():
        yield f"data: {json.dumps({'type': 'start', 'message': 'SSE连接开始'})}\n\n"
        time.sleep(1)
        yield f"data: {json.dumps({'type': 'content', 'content': 'Hello', 'done': False})}\n\n"
        time.sleep(1)
        yield f"data: {json.dumps({'type': 'content', 'content': ' World', 'done': False})}\n\n"
        time.sleep(1)
        yield f"data: {json.dumps({'type': 'content', 'content': '!', 'done': True})}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'message': 'SSE测试完成'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

@chat_bp.route('/messages/<int:message_id>/feedback', methods=['POST'])
@login_required
def update_message_feedback(message_id):
    """更新消息的反馈状态"""
    try:
        data = request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({'message': '缺少反馈参数'}), 400
            
        feedback = data['feedback']
        if feedback not in [1, -1]:
            return jsonify({'message': '无效的反馈值'}), 400
            
        message = chat_controller.get_message_by_id(message_id)
        if not message:
            return jsonify({'message': '消息不存在'}), 404
            
        # 检查消息是否属于当前用户的会话
        chat_session = ChatSession.query.get(message.session_id)
        if not chat_session or chat_session.user_id != session['user_id']:
            return jsonify({'message': '无权限操作此消息'}), 403
            
        # 使用 SQLAlchemy 实例的 session
        message.feedback_status = feedback
        db.session.commit()
        
        return jsonify({
            'message': '反馈更新成功',
            'feedback_status': feedback
        })
    except Exception as e:
        print(f"更新反馈失败：{str(e)}")
        db.session.rollback()  # 使用 SQLAlchemy 实例的 session
        return jsonify({'message': f'更新反馈失败：{str(e)}'}), 500

@chat_bp.route('/risk-assessment/<int:model_id>/form', methods=['GET'])
@login_required
def get_risk_assessment_form(model_id):
    """获取风险评估表单字段"""
    # 从请求参数中获取语言，默认为中文
    language = request.args.get('language', 'zh')
    fields = risk_config.get_form_fields(model_id, language=language)
    model_info = risk_config.get_model_info(model_id, language=language)
    
    if not fields or not model_info:
        return jsonify({'message': '未找到指定的风险评估模型'}), 404
        
    return jsonify({
        'fields': fields,
        'model_info': model_info
    })

@chat_bp.route('/risk-assessment/<int:model_id>/<int:session_id>/predict', methods=['POST'])
@login_required
def predict_risk(model_id, session_id):
    """风险评估预测"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': '请提供预测数据'}), 400
            
        # 获取模型信息
        model_info = risk_config.get_model_info(model_id)
        if not model_info:
            return jsonify({'message': '未找到指定的风险评估模型'}), 404
            
        # 获取表单字段定义
        fields = risk_config.get_form_fields(model_id)
        if not fields:
            return jsonify({'message': '未找到模型的表单定义'}), 404
            
        # 验证所有必需字段都已提供
        missing_fields = [field['name'] for field in fields if field['name'] not in data]
        if missing_fields:
            return jsonify({'message': f'缺少必需的字段: {", ".join(missing_fields)}'}), 400
        print(data,type(data))
        
        # 获取当前用户ID
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'message': '用户未登录'}), 401
        
        # 获取配置值
        risk_model_service_url = current_app.config['RISK_MODEL_SERVICE_URL']
        
        # 调用风险评估模型服务
        try:
            response = requests.post(
                f"{risk_model_service_url}/predict/{model_info['model_name']}",
                json=data,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            # 处理风险评估模型返回结果
            if model_info['model_type'] == 'regression':
                prediction = str(result['prediction'])
                prediction_label = prediction
                shap_html = result['shap_plot']
                probability = None
                confidence = None
            else:
                prediction = str(result['prediction'])
                prediction_label = result['prediction_label']
                probability = max(result['probabilities'].values())
                confidence = probability
                shap_html = result['shap_plot']
            
            # 获取语言参数（从JSON请求中获取，如果没有则默认为中文）
            language = data.get('language', 'zh')
            # 获取模型参数（从JSON请求中获取，如果没有则使用默认值）
            model = data.get('model', 'qwen3:32b')
            
            if model_info['model_type'] == 'regression':
                message = f"{'预测结果' if language == 'zh' else 'Prediction'}：{prediction}"
                nurse_response = get_nurse_response(model_info['model_name'], prediction, str(data), ollama_client, language, model=model)
            else:
                prob_text = f"{probability:.4f}" if probability is not None else ""
                message = (
                    f"预测结果：{prediction_label}，概率：{prob_text}"
                    if language == 'zh'
                    else f"Prediction: {prediction_label}, probability: {prob_text}"
                )
                nurse_response = get_nurse_response(model_info['model_name'], prediction_label, str(data), ollama_client, language, model=model)
            
            print(f"Nurse response: {nurse_response}")
            
            # 保存风险评估记录到数据库
            risk_assessment = RiskAssessment(
                user_id=user_id,
                session_id=session_id,
                model_id=model_id,
                model_name=model_info['model_name'],
                model_name_zh=model_info['model_name_zh'],
                form_data=data,
                prediction=prediction,
                prediction_label=prediction_label,
                probability=probability,
                confidence=confidence,
                shap_plot=shap_html,
                nurse_response=nurse_response,
                status='completed'
            )
            
            db.session.add(risk_assessment)
            db.session.commit()
            
            print(f"风险评估记录已保存到数据库，ID: {risk_assessment.id}")
            
            # 保存护士建议作为新消息
            nurse_message = chat_controller.add_message(
                session_id,
                nurse_response,
                is_user=False,
                message_type=0,
                is_visible=True
            )
            
            return jsonify({
                'prediction': prediction,
                'prediction_label': prediction_label,
                'probability': probability,
                'message': message,
                'nurse_response': nurse_response,
                'shap_html': shap_html,
                'assessment_id': risk_assessment.id,
                'messages': [{
                    'id': nurse_message.id,
                    'content': nurse_response,
                    'role': 'assistant',
                    'message_type': 0,
                    'feedback_status': 0
                }]
            })
        
            
        except requests.exceptions.RequestException as e:
            print(f"调用风险评估模型服务失败: {str(e)}")
            return jsonify({'message': '风险评估服务暂时不可用，请稍后重试'}), 503
            
    except Exception as e:
        print(f"风险预测过程中发生错误: {str(e)}")
        db.session.rollback()
        return jsonify({'message': f'预测失败：{str(e)}'}), 500

@chat_bp.route('/risk-assessment/models', methods=['GET'])
@login_required
def get_risk_assessment_models():
    """获取可用的风险评估模型列表"""
    try:
        risk_model_service_url = current_app.config['RISK_MODEL_SERVICE_URL']
        # 调用风险评估服务获取模型列表
        response = requests.get(f"{risk_model_service_url}/models")
        if response.status_code == 200:
            models = response.json()
            return jsonify(models)
        else:
            return jsonify({'error': '无法获取模型列表'}), 500
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败：{str(e)}'}), 500

@chat_bp.route('/risk-assessment/history', methods=['GET'])
@login_required
def get_risk_assessment_history():
    """获取用户的风险评估历史记录"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'message': '用户未登录'}), 401
        
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        session_id = request.args.get('session_id', type=int)
        model_id = request.args.get('model_id', type=int)
        
        # 构建查询
        query = RiskAssessment.query.filter_by(user_id=user_id)
        
        if session_id:
            query = query.filter_by(session_id=session_id)
        
        if model_id:
            query = query.filter_by(model_id=model_id)
        
        # 按创建时间倒序排列
        query = query.order_by(RiskAssessment.created_at.desc())
        
        # 分页
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        # 转换为字典格式
        assessments = [assessment.to_dict() for assessment in pagination.items]
        
        return jsonify({
            'assessments': assessments,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        })
        
    except Exception as e:
        print(f"获取风险评估历史记录失败: {str(e)}")
        return jsonify({'message': f'获取历史记录失败：{str(e)}'}), 500

@chat_bp.route('/risk-assessment/<int:assessment_id>', methods=['GET'])
@login_required
def get_risk_assessment_detail(assessment_id):
    """获取特定风险评估记录的详细信息"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'message': '用户未登录'}), 401
        
        # 查询风险评估记录
        assessment = RiskAssessment.query.filter_by(
            id=assessment_id, 
            user_id=user_id
        ).first()
        
        if not assessment:
            return jsonify({'message': '未找到指定的风险评估记录'}), 404
        
        return jsonify(assessment.to_dict())
        
    except Exception as e:
        print(f"获取风险评估详情失败: {str(e)}")
        return jsonify({'message': f'获取详情失败：{str(e)}'}), 500

@chat_bp.route('/risk-assessment/<int:assessment_id>', methods=['PUT'])
@login_required
def update_risk_assessment(assessment_id):
    """更新风险评估记录"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'message': '用户未登录'}), 401
        
        # 查询风险评估记录
        assessment = RiskAssessment.query.filter_by(
            id=assessment_id, 
            user_id=user_id
        ).first()
        
        if not assessment:
            return jsonify({'message': '未找到指定的风险评估记录'}), 404
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({'message': '无效的请求数据'}), 400
        
        # 获取模型信息
        model_id = assessment.model_id
        model_info = get_risk_model_info(model_id)
        if not model_info:
            return jsonify({'message': '风险模型不存在'}), 404
        
        # 调用风险预测服务
        prediction_response = requests.post(
            f"{RISK_MODEL_SERVICE_URL}/predict",
            json={
                'model_id': model_id,
                'data': data
            },
            timeout=30
        )
        
        if prediction_response.status_code != 200:
            return jsonify({'message': '风险预测服务调用失败'}), 500
        
        prediction_data = prediction_response.json()
        
        # 解析预测结果
        prediction = prediction_data.get('prediction', '')
        prediction_label = prediction_data.get('prediction_label', '')
        probability = prediction_data.get('probability')
        confidence = prediction_data.get('confidence')
        
        # 获取SHAP图表
        shap_html = None
        try:
            shap_response = requests.post(
                f"{RISK_MODEL_SERVICE_URL}/shap",
                json={
                    'model_id': model_id,
                    'data': data
                },
                timeout=30
            )
            if shap_response.status_code == 200:
                shap_data = shap_response.json()
                shap_html = shap_data.get('shap_html')
        except Exception as e:
            print(f"获取SHAP图表失败: {str(e)}")
        
        # 生成护士建议
        nurse_response = generate_nurse_response(
            model_info['model_name_zh'],
            prediction,
            prediction_label,
            probability,
            data
        )
        
        # 更新记录
        assessment.form_data = data
        assessment.prediction = prediction
        assessment.prediction_label = prediction_label
        assessment.probability = probability
        assessment.confidence = confidence
        assessment.shap_plot = shap_html
        assessment.nurse_response = nurse_response
        assessment.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        print(f"风险评估记录已更新，ID: {assessment.id}")
        
        # 返回更新后的结果
        return jsonify({
            'message': '评估更新成功',
            'assessment_id': assessment.id,
            'prediction': prediction,
            'prediction_label': prediction_label,
            'probability': probability,
            'confidence': confidence,
            'shap_html': shap_html,
            'nurse_response': nurse_response
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"更新风险评估记录失败: {str(e)}")
        return jsonify({'message': f'更新失败：{str(e)}'}), 500

@chat_bp.route('/risk-assessment/<int:assessment_id>', methods=['DELETE'])
@login_required
def delete_risk_assessment(assessment_id):
    """删除风险评估记录"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'message': '用户未登录'}), 401
        
        # 查询风险评估记录
        assessment = RiskAssessment.query.filter_by(
            id=assessment_id, 
            user_id=user_id
        ).first()
        
        if not assessment:
            return jsonify({'message': '未找到指定的风险评估记录'}), 404
        
        # 删除记录
        db.session.delete(assessment)
        db.session.commit()
        
        return jsonify({'message': '风险评估记录已删除'})
        
    except Exception as e:
        print(f"删除风险评估记录失败: {str(e)}")
        db.session.rollback()
        return jsonify({'message': f'删除失败：{str(e)}'}), 500

import markdown
from datetime import datetime

def render_health_report_html(
    report_md: str,
    title: str = "济世 · 报告解读",
    user_name: str = "用户上传",
    version: str = "v1.0",
    language: str = "zh",
) -> str:
    html_content = markdown.markdown(
        report_md,
        extensions=['tables', 'fenced_code']
    ).replace('<!--pagebreak-->', '<div class="page-break"></div>')
    
    # 获取logo图片并转换为base64编码
    # route.py 位于: client/app/modules/chat/route.py
    # logo.png 位于: client/app/static/images/logo.png
    # 需要向上4级到client目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    logo_path = os.path.join(base_dir, 'app', 'static', 'images', 'logo.png')
    logo_absolute_path = os.path.abspath(logo_path)
    
    # 读取logo图片并转换为base64
    logo_base64 = ""
    if os.path.exists(logo_absolute_path):
        try:
            with open(logo_absolute_path, 'rb') as f:
                logo_data = f.read()
                logo_base64 = base64.b64encode(logo_data).decode('utf-8')
                logo_base64 = f"data:image/png;base64,{logo_base64}"
                print(f"Logo loaded successfully, base64 length: {len(logo_base64)}")
        except Exception as e:
            print(f"Error loading logo: {str(e)}")
            logo_base64 = ""
    else:
        print(f"Logo file not found at: {logo_absolute_path}")

    # 根据语言设置字体和冒号
    if language.lower() == 'en':
        font_family = 'Arial,Helvetica,"Segoe UI",Roboto,sans-serif'
        colon_char = ":"
    else:
        font_family = '"Noto Sans SC","PingFang SC","Microsoft YaHei",Arial,Helvetica,sans-serif'
        colon_char = "："
    
    style_css = f"""
:root {{ --primary:#1976d2; --ok:#2e7d32; --warn:#ef6c00; --danger:#d32f2f;
        --border:#e5e7eb; --muted:#6b7280; --bg:#ffffff; }}
* {{ box-sizing:border-box; }}
html,body {{ height:100%; }}
body {{ font-family:{font_family};
       margin:0; background:#f6f7f9; color:#111827; line-height:1.65; }}
.page {{ width:210mm; min-height:297mm; margin:0 auto; background:var(--bg);
        padding:4mm 4mm 8mm 4mm; box-shadow:0 6px 30px rgba(0,0,0,.08); }}

/* ===== 顶部三行布局 ===== */
header.report-header{{
  display:flex;
  flex-direction:column;
  align-items:stretch;
  row-gap:6px;
  margin-bottom:4mm;
  border-bottom:1px solid var(--border);
  padding-bottom:3mm;
}}

/* 行1：左上 logo（含文字） */
.brand{{
  align-self:flex-start;
  display:inline-flex; align-items:center; gap:10px;
  font-weight:700; font-size:20px; color:var(--primary);
  white-space:nowrap; word-break:keep-all; line-height:1;
}}
.brand .logo{{ width:40px; height:40px; display:inline-block; object-fit:contain; }}

/* 行2：标题加粗居中 */
.report-title{{
  align-self:center; text-align:center; width:100%;
  margin:0; font-size:22px; font-weight:700; color:#0f172a;
  white-space:nowrap;
}}

/* 行3：信息条（单个框体，内容居中） */
.meta-box{{
  align-self:center;
  background:#fff; border:1px solid var(--border); border-radius:14px;
  padding:12px 20px; width:fit-content; min-width:80%;
  margin:0 auto;
}}
.meta-items{{
  display:flex; align-items:center; justify-content:center;
  gap:14px; flex-wrap:nowrap;
}}
.meta-item{{ display:inline-flex; align-items:center; gap:6px; white-space:nowrap; flex-shrink:0; }}
.meta-item + .meta-item{{
  position:relative; padding-left:14px; margin-left:0;
}}
.meta-item + .meta-item::before{{
  content:"|"; position:absolute; left:0; top:50%; transform:translateY(-50%);
  color:#cbd5e1;
}}
.meta-item .k{{ color:var(--muted); }}
.meta-item .k::after{{ content:"{colon_char}"; margin:0 2px; color:var(--muted); }}
.meta-item .v{{ font-weight:600; color:#0f172a; }}

/* 正文与表格等 */
h1,h2,h3 {{ margin:12px 0 8px; line-height:1.3; }}
h1 {{ text-align:center; color:var(--primary); font-size:26px; margin-top:1mm; }}
h2 {{ font-size:18px; border-left:4px solid var(--primary); padding-left:8px; }}
h3 {{ font-size:15px; color:#0f172a; }}
p {{ margin:6px 0; max-width:100%; }}
table {{ width:100%; border-collapse:separate; border-spacing:0; margin:8px 0 12px; max-width:100%; }}
thead th {{ background:#f8fafc; position:sticky; top:0; z-index:1; }}
th,td {{ border:1px solid var(--border); padding:8px; text-align:left; font-size:13px; }}
tbody tr:nth-child(odd) td {{ background:#fcfcfd; }}
tbody tr:hover td {{ background:#f5f9ff; }}
.callout{{ border-left:4px solid var(--warn); background:#fff7ed; padding:10px 12px; border-radius:8px; margin:10px 0; }}
.tag{{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid var(--border); }}
.tag.ok{{ background:#eef7ee; border-color:#d7ebd7; color:var(--ok); }}
.tag.high{{ background:#fff0f0; border-color:#f4d6d6; color:var(--danger); }}
.tag.low{{ background:#fff8e1; border-color:#ffecb3; color:#a65d00; }}
hr{{ border:none; border-top:1px solid var(--border); margin:14px 0; }}
.disclaimer{{ font-size:12px; color:#6b7280; margin-top:4mm; }}
footer.report-footer{{
  margin-top:6mm; padding-top:4mm; border-top:1px solid var(--border); color:#6b7280; font-size:12px;
  display:flex; align-items:center; justify-content:space-between;
}}
.page-break{{ break-before:page; page-break-before:always; height:0; }}
@media print{{
  body{{ background:#fff; }}
  .page{{ width:auto; min-height:auto; box-shadow:none; padding:8mm 4mm; }}
  a[href]::after{{ content:""; }}
  thead{{ display:table-header-group; }}
  tr,img{{ break-inside:avoid; page-break-inside:avoid; }}
  .no-print{{ display:none !important; }}
}}
"""

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    year = datetime.now().year
    
    # 根据语言设置文本内容
    if language.lower() == 'en':
        # 英文版本
        lang_code = "en-US"
        brand_name = "JiShi"
        default_title = "JiShi · Report"
        if title == "济世 · 报告解读":
            title = default_title
        meta_labels = {
            "generated_time": "Generated Time",
            "version": "Version",
            "source": "Source"
        }
        default_user_name = "User Upload"
        if user_name == "用户上传":
            user_name = default_user_name
        disclaimer_text = f'This report is automatically generated by the intelligent agent <strong>JiShi</strong> based on the information you provided, for health reference only and not as a basis for clinical diagnosis. If there are any abnormalities, it is recommended to go to a formal medical institution for further examination and consultation.'
        footer_left = f"© {year} JiShi · AI Health Management Assistant"
        footer_right = 'It is recommended to export as PDF (A4 portrait) by clicking "Print → Save as PDF" in the upper right corner.'
        print_button = "Export as PDF"
    else:
        # 中文版本
        lang_code = "zh-CN"
        brand_name = "济世"
        default_title = "济世 · 报告解读"
        if title == "济世 · 报告解读":
            title = default_title
        meta_labels = {
            "generated_time": "生成时间",
            "version": "版本",
            "source": "来源"
        }
        default_user_name = "用户上传"
        if user_name == "用户上传":
            user_name = default_user_name
        disclaimer_text = f'本报告由智能体<strong>济世</strong>基于您提供的信息自动生成，仅供健康参考，不作为临床诊断依据。如有异常建议至正规医疗机构进一步检查与就诊。'
        footer_left = f"© {year} 济世 · AI 健康管理助手"
        footer_right = '建议按右上角"打印 → 另存为PDF"导出为 PDF（A4 纵向）。'
        print_button = "导出为 PDF"

    styled_html = f"""<!doctype html>
<html lang="{lang_code}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>{style_css}</style>
</head>
<body>
  <div class="page">
    <header class="report-header">
      <!-- 行1：左上 logo -->
      <div class="brand">
        {'<img src="' + logo_base64 + '" class="logo" alt="Logo">' if logo_base64 else ''}<span>{brand_name}</span>
      </div>

      <!-- 行2：居中标题（加粗） -->
      <div class="report-title">{title}</div>

      <!-- 行3：居中信息条 -->
      <div class="meta-box">
        <div class="meta-items">
          <div class="meta-item"><span class="k">{meta_labels['generated_time']}</span><span class="v">{now_str}</span></div>
          <div class="meta-item"><span class="k">{meta_labels['version']}</span><span class="v">{version}</span></div>
          <div class="meta-item"><span class="k">{meta_labels['source']}</span><span class="v">{user_name}</span></div>
        </div>
      </div>
    </header>

    {html_content}

    <div class="disclaimer">
      {disclaimer_text}
    </div>

    <footer class="report-footer">
      <span>{footer_left}</span>
      <span>{footer_right}</span>
    </footer>
  </div>

  <button class="no-print" onclick="window.print()" style="position:fixed;right:16px;bottom:16px;border:none;padding:10px 14px;border-radius:10px;background:var(--primary);color:#fff;cursor:pointer;">{print_button}</button>
</body>
</html>
"""
    return styled_html







@chat_bp.route('/sessions/<int:session_id>/export', methods=['GET'])
@login_required
def export_report(session_id):
    """导出健康体检报告"""
    try:
        # 获取会话的所有消息
        messages = chat_controller.get_session_messages(session_id)
        
        # 将消息格式化为对话形式
        dialogue = ""
        for msg in messages:
            role = "用户" if msg.is_user else "医生"
            dialogue += f"{role}：{msg.content}\n"
        print("当前会话对话信息：",dialogue)
        # 获取语言参数（从请求参数中获取，如果没有则默认为中文）
        language = request.args.get('language', 'zh')
        # 获取模型参数（从请求参数中获取，如果没有则使用默认值）
        model = request.args.get('model', 'qwen3:32b')
        # 生成健康体检报告
        report_result = generate_health_report(dialogue, ollama_client, language, model=model)
        
        # 处理返回的报告结果
        if isinstance(report_result, dict):
            report = report_result.get("llm_response", "抱歉，暂时无法生成健康体检报告。请稍后再试。")
        else:
            report = report_result
        
        # 去除<think></think>部分
        from app.util.RAG import clean_think
        report = clean_think(report)
        
        # 去除所有的```标记
        import re
        report = re.sub(r'```[a-zA-Z]*\s*', '', report)
        report = re.sub(r'\s*```', '', report)
        
        print(report)   
        # 将报告转换为PDF
        try:
            import markdown
            from weasyprint import HTML
            import tempfile
            import os
            from datetime import datetime

            styled_html = render_health_report_html(report, language=language)
            
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"health_report_{timestamp}.pdf"
            pdf_path = os.path.join(temp_dir, pdf_filename)
            
            # 生成PDF
            HTML(string=styled_html).write_pdf(pdf_path)
            
            # 发送PDF文件
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype='application/pdf'
            )
            
        except Exception as e:
            print(f"Error converting report to PDF: {str(e)}")
            # 如果转换失败，返回Markdown格式的报告
            return jsonify({
                'report': report,
                'format': 'markdown'
            })
            
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({'message': '生成报告失败，请稍后重试'}), 500 

@chat_bp.route('/messages/<int:message_id>/image', methods=['GET'])
@login_required
def get_message_image(message_id):
    message = chat_controller.get_message_by_id(message_id)
    if not message or not message.has_image:
        return jsonify({'error': 'No image'}), 404
    return jsonify({'image_data': message.image_data}) 