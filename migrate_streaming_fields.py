#!/usr/bin/env python3
"""
数据库迁移脚本 - 为 chat_messages 表添加流式响应相关字段
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import create_app
from app.db.db import db
from sqlalchemy import inspect, text

def migrate_streaming_fields():
    """为 chat_messages 表添加流式响应相关字段"""
    app = create_app('development')
    
    with app.app_context():
        try:
            # 检查表是否存在
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            if 'chat_messages' not in existing_tables:
                print("❌ chat_messages 表不存在，请先创建表")
                return False
            
            # 检查字段是否已存在
            columns = [col['name'] for col in inspector.get_columns('chat_messages')]
            new_fields = {
                'is_streaming': 'BOOLEAN DEFAULT 0',
                'streaming_content': 'TEXT',
                'streaming_think_content': 'TEXT',
                'streaming_updated_at': 'DATETIME'
            }
            
            fields_to_add = []
            for field_name, field_type in new_fields.items():
                if field_name not in columns:
                    fields_to_add.append((field_name, field_type))
                    print(f"需要添加字段: {field_name}")
                else:
                    print(f"字段 {field_name} 已存在，跳过")
            
            if not fields_to_add:
                print("✅ 所有字段已存在，无需迁移")
                return True
            
            # 添加新字段
            print(f"\n正在添加 {len(fields_to_add)} 个新字段...")
            with db.engine.connect() as conn:
                for field_name, field_type in fields_to_add:
                    try:
                        # SQLite 不支持在一条 ALTER TABLE 中添加多个列，需要分别执行
                        sql = f"ALTER TABLE chat_messages ADD COLUMN {field_name} {field_type}"
                        conn.execute(text(sql))
                        conn.commit()
                        print(f"✅ 成功添加字段: {field_name}")
                    except Exception as e:
                        # 如果字段已存在（可能在其他地方已添加），忽略错误
                        if 'duplicate column' in str(e).lower() or 'already exists' in str(e).lower():
                            print(f"⚠️  字段 {field_name} 可能已存在，跳过")
                        else:
                            print(f"❌ 添加字段 {field_name} 失败: {str(e)}")
                            return False
            
            # 验证字段是否添加成功
            inspector = inspect(db.engine)
            columns_after = [col['name'] for col in inspector.get_columns('chat_messages')]
            
            all_added = True
            for field_name, _ in fields_to_add:
                if field_name not in columns_after:
                    print(f"❌ 字段 {field_name} 添加失败")
                    all_added = False
            
            if all_added:
                print("\n✅ 所有字段添加成功！")
                print(f"迁移时间: {datetime.now()}")
                print("\n添加的字段:")
                for field_name, field_type in fields_to_add:
                    print(f"  - {field_name}: {field_type}")
                return True
            else:
                print("\n❌ 部分字段添加失败")
                return False
                
        except Exception as e:
            print(f"❌ 迁移失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("开始数据库迁移 - 添加流式响应字段")
    print("=" * 60)
    success = migrate_streaming_fields()
    if success:
        print("\n" + "=" * 60)
        print("✅ 迁移完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 迁移失败！")
        print("=" * 60)
        sys.exit(1)

