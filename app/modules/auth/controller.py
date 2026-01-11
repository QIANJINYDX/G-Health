from werkzeug.security import generate_password_hash, check_password_hash
from app.db.models import User, ChatSession, ChatMessage, RiskAssessment
from app.db.db import db
from datetime import datetime, timedelta
from sqlalchemy import func

class AuthController:
    def login(self, username, password):
        """用户登录"""
        if not username or not password:
            return {'success': False, 'message': '用户名和密码不能为空'}
            
        user = User.query.filter_by(username=username).first()
        if not user:
            return {'success': False, 'message': '用户名或密码错误'}
            
        if not check_password_hash(user.password_hash, password):
            return {'success': False, 'message': '用户名或密码错误'}
            
        # 更新最后登录时间
        try:
            from datetime import datetime
            user.last_login = datetime.utcnow()
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            # 登录时间更新失败不影响登录流程
            
        return {'success': True, 'user_id': user.id}
        
    def register(self, username, password):
        """用户注册"""
        if not username or not password:
            return {'success': False, 'message': '用户名和密码不能为空'}
            
        if User.query.filter_by(username=username).first():
            return {'success': False, 'message': '用户名已存在'}
            
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            return {'success': True}
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': '注册失败，请稍后重试'}

    def get_user_info(self, user_id):
        """获取用户信息"""
        user = User.query.get(user_id)
        if not user:
            return {'success': False, 'message': '用户不存在'}
            
        return {
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'updated_at': user.updated_at.isoformat() if user.updated_at else None,
                'last_login': user.last_login.isoformat() if user.last_login else None
            }
        }

    def change_password(self, user_id, new_password):
        """修改密码（不需要验证当前密码）"""
        if not new_password:
            return {'success': False, 'message': '新密码不能为空'}
            
        user = User.query.get(user_id)
        if not user:
            return {'success': False, 'message': '用户不存在'}
            
        try:
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            return {'success': True}
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': '修改失败，请稍后重试'}

    def change_password_with_verification(self, user_id, current_password, new_password):
        """修改密码（需要验证当前密码）"""
        if not current_password or not new_password:
            return {'success': False, 'message': '当前密码和新密码不能为空'}
            
        user = User.query.get(user_id)
        if not user:
            return {'success': False, 'message': '用户不存在'}
            
        # 验证当前密码
        if not check_password_hash(user.password_hash, current_password):
            return {'success': False, 'message': '当前密码错误'}
            
        try:
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            return {'success': True}
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': '修改失败，请稍后重试'}

    def get_user_stats(self, user_id):
        """获取用户使用统计"""
        user = User.query.get(user_id)
        if not user:
            return {'success': False, 'message': '用户不存在'}
            
        try:
            # 统计总对话数
            total_chats = len(user.chat_sessions)
            
            # 统计总消息数
            total_messages = 0
            for session in user.chat_sessions:
                total_messages += len(session.messages)
            
            # 统计风险评估数（message_type=3的消息）
            total_assessments = 0
            for session in user.chat_sessions:
                for message in session.messages:
                    if message.message_type == 3:
                        total_assessments += 1
            
            # 统计导出报告数（这里暂时设为0，因为还没有实现报告导出统计）
            total_reports = 0
            
            return {
                'success': True,
                'stats': {
                    'total_chats': total_chats,
                    'total_messages': total_messages,
                    'total_assessments': total_assessments,
                    'total_reports': total_reports
                }
            }
        except Exception as e:
            return {'success': False, 'message': '获取统计信息失败'}

    def get_chat_stats_by_date(self, user_id, days=30):
        """获取按日期分组的对话统计
        Args:
            user_id: 用户ID
            days: 统计最近多少天的数据，默认30天
        Returns:
            {
                'success': True,
                'data': [
                    {'date': '2025-01-01', 'count': 5},
                    {'date': '2025-01-02', 'count': 3},
                    ...
                ]
            }
        """
        user = User.query.get(user_id)
        if not user:
            return {'success': False, 'message': '用户不存在'}
        
        try:
            # 计算起始日期
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days-1)
            
            # 查询按日期分组的对话数
            # 使用 func.date() 来提取日期部分
            stats = db.session.query(
                func.date(ChatSession.created_at).label('date'),
                func.count(ChatSession.id).label('count')
            ).filter(
                ChatSession.user_id == user_id,
                func.date(ChatSession.created_at) >= start_date,
                func.date(ChatSession.created_at) <= end_date
            ).group_by(
                func.date(ChatSession.created_at)
            ).order_by(
                func.date(ChatSession.created_at)
            ).all()
            
            # 转换为字典列表
            result = [{'date': str(stat.date), 'count': stat.count} for stat in stats]
            
            # 填充缺失的日期（确保连续的时间序列）
            date_list = []
            current_date = start_date
            while current_date <= end_date:
                date_str = str(current_date)
                # 查找该日期的统计数据
                stat = next((s for s in result if s['date'] == date_str), None)
                if stat:
                    date_list.append(stat)
                else:
                    date_list.append({'date': date_str, 'count': 0})
                current_date += timedelta(days=1)
            
            return {
                'success': True,
                'data': date_list
            }
        except Exception as e:
            return {'success': False, 'message': f'获取日期统计失败: {str(e)}'}

    def get_message_stats_by_date(self, user_id, days=30):
        """获取按日期分组的消息统计
        Args:
            user_id: 用户ID
            days: 统计最近多少天的数据，默认30天
        Returns:
            {
                'success': True,
                'data': [
                    {'date': '2025-01-01', 'count': 15},
                    {'date': '2025-01-02', 'count': 8},
                    ...
                ]
            }
        """
        user = User.query.get(user_id)
        if not user:
            return {'success': False, 'message': '用户不存在'}
        
        try:
            # 计算起始日期
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days-1)
            
            # 查询按日期分组的消息数（通过会话关联）
            stats = db.session.query(
                func.date(ChatMessage.created_at).label('date'),
                func.count(ChatMessage.id).label('count')
            ).join(
                ChatSession, ChatMessage.session_id == ChatSession.id
            ).filter(
                ChatSession.user_id == user_id,
                func.date(ChatMessage.created_at) >= start_date,
                func.date(ChatMessage.created_at) <= end_date
            ).group_by(
                func.date(ChatMessage.created_at)
            ).order_by(
                func.date(ChatMessage.created_at)
            ).all()
            
            # 转换为字典列表
            result = [{'date': str(stat.date), 'count': stat.count} for stat in stats]
            
            # 填充缺失的日期（确保连续的时间序列）
            date_list = []
            current_date = start_date
            while current_date <= end_date:
                date_str = str(current_date)
                # 查找该日期的统计数据
                stat = next((s for s in result if s['date'] == date_str), None)
                if stat:
                    date_list.append(stat)
                else:
                    date_list.append({'date': date_str, 'count': 0})
                current_date += timedelta(days=1)
            
            return {
                'success': True,
                'data': date_list
            }
        except Exception as e:
            return {'success': False, 'message': f'获取消息统计失败: {str(e)}'}

    def get_risk_assessment_stats_by_date(self, user_id, days=30):
        """获取按日期分组的风险评估统计
        Args:
            user_id: 用户ID
            days: 统计最近多少天的数据，默认30天
        Returns:
            {
                'success': True,
                'data': [
                    {'date': '2025-01-01', 'count': 3},
                    {'date': '2025-01-02', 'count': 2},
                    ...
                ]
            }
        """
        user = User.query.get(user_id)
        if not user:
            return {'success': False, 'message': '用户不存在'}
        
        try:
            # 计算起始日期
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days-1)
            
            # 查询按日期分组的风险评估数
            stats = db.session.query(
                func.date(RiskAssessment.created_at).label('date'),
                func.count(RiskAssessment.id).label('count')
            ).filter(
                RiskAssessment.user_id == user_id,
                func.date(RiskAssessment.created_at) >= start_date,
                func.date(RiskAssessment.created_at) <= end_date
            ).group_by(
                func.date(RiskAssessment.created_at)
            ).order_by(
                func.date(RiskAssessment.created_at)
            ).all()
            
            # 转换为字典列表
            result = [{'date': str(stat.date), 'count': stat.count} for stat in stats]
            
            # 填充缺失的日期（确保连续的时间序列）
            date_list = []
            current_date = start_date
            while current_date <= end_date:
                date_str = str(current_date)
                # 查找该日期的统计数据
                stat = next((s for s in result if s['date'] == date_str), None)
                if stat:
                    date_list.append(stat)
                else:
                    date_list.append({'date': date_str, 'count': 0})
                current_date += timedelta(days=1)
            
            return {
                'success': True,
                'data': date_list
            }
        except Exception as e:
            return {'success': False, 'message': f'获取风险评估统计失败: {str(e)}'} 