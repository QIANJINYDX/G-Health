from app.db.models import ChatSession, ChatMessage
from app.db.db import db

class ChatController:
    def create_session(self, user_id, title=None):
        """创建新的聊天会话"""
        session = ChatSession(
            user_id=user_id,
            title=title or "新对话"
        )
        db.session.add(session)
        db.session.commit()
        return session

    def get_user_sessions(self, user_id):
        """获取用户的所有聊天会话"""
        return ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).all()

    def get_session_messages(self, session_id,vis=True):
        """获取会话的所有可见消息"""
        if vis:
            return ChatMessage.query.filter_by(session_id=session_id, is_visible=vis).order_by(ChatMessage.created_at).all()
        else:
            return ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at).all()
    
    def get_session_messages_by_type(self, session_id, message_type):
        """获取会话的指定类型的消息"""
        return ChatMessage.query.filter_by(session_id=session_id, message_type=message_type).order_by(ChatMessage.created_at).all()

    def add_message(self, session_id, content, is_user=True, message_type=0, is_visible=True, risk_model=None, references=None, mcp_response=None, has_image=False, image_data=None, follow_up_questions=None, display_mode='default', stages=None, is_streaming=False):
        """添加新消息"""
        message = ChatMessage(
            session_id=session_id,
            content=content,
            is_user=is_user,
            message_type=message_type,
            is_visible=is_visible,
            risk_model=risk_model,
            references=references,
            mcp_response=mcp_response,
            has_image=has_image,
            image_data=image_data,
            follow_up_questions=follow_up_questions,
            display_mode=display_mode,
            stages=stages,
            is_streaming=is_streaming,
            streaming_content=content if is_streaming else None
        )
        db.session.add(message)
        
        # 更新会话标题（使用第一条用户消息）
        session = ChatSession.query.get(session_id)
        if is_user and not session.title or session.title == "新对话":
            session.title = content[:20] + ("..." if len(content) > 20 else "")
        
        db.session.commit()
        return message
    
    def update_streaming_content(self, message_id, content, think_content=None):
        """更新流式消息的内容"""
        from datetime import datetime
        message = ChatMessage.query.get(message_id)
        if message:
            message.streaming_content = content
            message.content = content  # 同时更新content字段
            if think_content is not None:
                message.streaming_think_content = think_content
            message.streaming_updated_at = datetime.utcnow()
            db.session.commit()
            return True
        return False
    
    def complete_streaming(self, message_id, final_content=None, follow_up_questions=None):
        """完成流式响应"""
        from datetime import datetime
        message = ChatMessage.query.get(message_id)
        if message:
            message.is_streaming = False
            if final_content:
                message.content = final_content
                message.streaming_content = None
            else:
                # 如果没有提供最终内容，使用流式内容
                if message.streaming_content:
                    message.content = message.streaming_content
                    message.streaming_content = None
            if follow_up_questions:
                message.follow_up_questions = follow_up_questions
            message.streaming_updated_at = datetime.utcnow()
            db.session.commit()
            return True
        return False
    
    def get_streaming_message(self, session_id):
        """获取会话中正在流式生成的消息"""
        return ChatMessage.query.filter_by(
            session_id=session_id,
            is_streaming=True,
            is_user=False
        ).order_by(ChatMessage.created_at.desc()).first()

    def delete_session(self, session_id):
        """删除聊天会话"""
        session = ChatSession.query.get(session_id)
        if session:
            # 删除所有相关消息
            ChatMessage.query.filter_by(session_id=session_id).delete()
            db.session.delete(session)
            db.session.commit()
            return True
        return False

    def get_message_by_id(self, message_id):
        """根据ID获取消息"""
        return ChatMessage.query.get(message_id) 