#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG文件清理工具
用于清理markdown格式的RAG文件，去除目录页、免责声明、版权声明、参考文献、URL链接等
"""

import os
import re
from pathlib import Path
from typing import List


class RAGFileCleaner:
    """RAG文件清理器"""
    
    def __init__(self, target_dir: str):
        """
        初始化清理器
        
        Args:
            target_dir: 目标目录路径
        """
        self.target_dir = Path(target_dir)
        if not self.target_dir.exists():
            raise ValueError(f"目录不存在: {target_dir}")
    
    def clean_file(self, file_path: Path) -> str:
        """
        清理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            清理后的内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. 去除目录页
        content = self._remove_table_of_contents(content)
        
        # 2. 去除免责声明、版权声明
        content = self._remove_disclaimers(content)
        
        # 3. 去除参考文献区
        content = self._remove_references(content)
        
        # 4. 去除URL链接
        content = self._remove_urls(content)
        
        # 5. 去除HTML标签
        content = self._remove_html_tags(content)
        
        # 6. 清理无意义空行和连续空格
        content = self._clean_whitespace(content)
        
        return content
    
    def _remove_table_of_contents(self, content: str) -> str:
        """
        去除目录页
        
        识别以"目录"、"Contents"等开头的章节，并删除到下一个一级标题之前的内容
        """
        lines = content.split('\n')
        result = []
        skip_mode = False
        
        # 目录相关的关键词
        toc_keywords = ['目录', 'Contents', 'CONTENTS', '目  录', '目　录']
        
        for i, line in enumerate(lines):
            # 检查是否是目录标题（支持带数字编号，注意支持中文全角符号"．"）
            if re.match(r'^#+\s*[\d\.．]*\s*(目录|Contents|CONTENTS|目\s*录)', line, re.IGNORECASE):
                skip_mode = True
                continue
            
            # 如果遇到下一个一级标题（# 开头），停止跳过
            if skip_mode and re.match(r'^#\s+[^#]', line):
                skip_mode = False
                result.append(line)
            elif not skip_mode:
                result.append(line)
        
        return '\n'.join(result)
    
    def _remove_disclaimers(self, content: str) -> str:
        """
        去除免责声明、版权声明
        
        识别包含"免责"、"版权"、"Copyright"等关键词的段落
        """
        lines = content.split('\n')
        result = []
        skip_mode = False
        
        # 免责声明和版权声明的关键词
        disclaimer_keywords = [
            '免责', '免责声明', '免责条款',
            '版权', '版权声明', '版权所有', 'Copyright', '©',
            '声明', 'Disclaimer', 'DISCLAIMER'
        ]
        
        for i, line in enumerate(lines):
            # 检查是否包含免责/版权关键词
            if any(keyword in line for keyword in disclaimer_keywords):
                # 检查是否是标题行
                if re.match(r'^#+\s*', line):
                    skip_mode = True
                    continue
                # 检查是否是普通行但包含关键词
                elif any(keyword in line for keyword in disclaimer_keywords):
                    skip_mode = True
                    continue
            
            # 如果遇到下一个一级或二级标题，停止跳过
            if skip_mode and re.match(r'^#{1,2}\s+[^#]', line):
                skip_mode = False
                result.append(line)
            elif not skip_mode:
                result.append(line)
        
        return '\n'.join(result)
    
    def _remove_references(self, content: str) -> str:
        """
        去除参考文献区
        
        识别以"参考文献"、"References"等开头的章节，并删除到文件末尾或下一个一级标题之前的内容
        """
        lines = content.split('\n')
        result = []
        skip_mode = False
        
        # 参考文献相关的关键词
        ref_keywords = [
            '参考文献', '参考', 'References', 'REFERENCES',
            '文献', '典型文献', '引用文献', 'Bibliography'
        ]
        
        for i, line in enumerate(lines):
            # 检查是否是参考文献标题（支持带数字编号，如 "# 3.参考文献：" 或 "# 5．参考文献1"）
            # 注意：支持中文全角符号"．"和英文句号"."
            if re.match(r'^#+\s*[\d\.．]*\s*(参考文献|参考|References|REFERENCES|文献|典型文献|引用文献|Bibliography)[\d\s]*', line, re.IGNORECASE):
                skip_mode = True
                continue
            
            # 如果遇到下一个一级标题（# 开头），停止跳过
            if skip_mode:
                if re.match(r'^#\s+[^#]', line):
                    skip_mode = False
                    result.append(line)
                # 否则继续跳过
                continue
            
            # 正常添加行
            result.append(line)
        
        return '\n'.join(result)
    
    def _remove_urls(self, content: str) -> str:
        """
        去除URL链接
        
        去除所有HTTP/HTTPS链接，以及常见的URL模式
        """
        lines = content.split('\n')
        result = []
        
        for line in lines:
            # 如果整行只包含"解螺旋"和URL，则跳过这一行
            if re.match(r'^\s*解螺旋\s*(https?://|www\.)', line):
                continue
            
            # 移除行中的URL（保留其他内容）
            # 移除 "解螺旋 http://..." 或 "解螺旋 https://..." 模式
            line = re.sub(r'解螺旋\s*https?://[^\s\)\]\}]+', '', line)
            line = re.sub(r'解螺旋\s*www\.[^\s\)\]\}]+', '', line)
            
            # 移除 http:// 或 https:// 开头的URL
            line = re.sub(r'https?://[^\s\)\]\}]+', '', line)
            
            # 移除 www. 开头的URL
            line = re.sub(r'www\.[^\s\)\]\}]+', '', line)
            
            # 清理行尾的"解螺旋"文本（如果URL已被移除但"解螺旋"还在）
            line = re.sub(r'解螺旋\s*$', '', line)
            line = re.sub(r'解螺旋\s*\.', '.', line)  # 处理"解螺旋."的情况
            
            result.append(line)
        
        return '\n'.join(result)
    
    def _remove_html_tags(self, content: str) -> str:
        """
        去除HTML标签
        
        去除所有HTML标签，但保留标签内的文本内容
        """
        # 去除完整的HTML标签对，如 <html><body><table>...</table></body></html>
        content = re.sub(r'<html>.*?</html>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<body>.*?</body>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<table>.*?</table>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # 去除所有HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        
        return content
    
    def _clean_whitespace(self, content: str) -> str:
        """
        清理无意义空行和连续空格
        
        1. 将多个连续空格替换为单个空格
        2. 将多个连续空行替换为最多两个空行
        3. 去除行首行尾的空格
        """
        # 去除行首行尾空格
        lines = [line.rstrip() for line in content.split('\n')]
        
        # 将多个连续空格替换为单个空格（但保留代码块中的空格）
        cleaned_lines = []
        for line in lines:
            # 将多个连续空格替换为单个空格
            line = re.sub(r' {2,}', ' ', line)
            cleaned_lines.append(line)
        
        # 清理空行：将3个或更多连续空行替换为2个空行
        content = '\n'.join(cleaned_lines)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 去除文件开头和结尾的空行
        content = content.strip()
        
        return content
    
    def clean_all_files(self, backup: bool = True) -> dict:
        """
        清理目录下所有md文件
        
        Args:
            backup: 是否备份原文件（在文件名后加.bak）
            
        Returns:
            清理结果统计
        """
        md_files = list(self.target_dir.glob('*.md'))
        
        stats = {
            'total': len(md_files),
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        for md_file in md_files:
            try:
                # 备份原文件
                if backup:
                    backup_path = md_file.with_suffix('.md.bak')
                    if not backup_path.exists():
                        import shutil
                        shutil.copy2(md_file, backup_path)
                
                # 清理文件
                cleaned_content = self.clean_file(md_file)
                
                # 写回文件
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                stats['success'] += 1
                print(f"✓ 已清理: {md_file.name}")
                
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append({
                    'file': md_file.name,
                    'error': str(e)
                })
                print(f"✗ 清理失败: {md_file.name} - {str(e)}")
        
        return stats


def main():
    """主函数"""
    # 目标目录
    target_dir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/MedicalExaminationAgent/PhysicalExaminationAgent/client/app/util/rag_data/md_files"
    
    # 创建清理器
    cleaner = RAGFileCleaner(target_dir)
    
    # 执行清理
    print(f"开始清理目录: {target_dir}")
    print(f"找到 {len(list(Path(target_dir).glob('*.md')))} 个md文件\n")
    
    stats = cleaner.clean_all_files(backup=False)
    
    # 打印统计信息
    print("\n" + "="*50)
    print("清理完成！")
    print(f"总计: {stats['total']} 个文件")
    print(f"成功: {stats['success']} 个文件")
    print(f"失败: {stats['failed']} 个文件")
    
    if stats['errors']:
        print("\n错误详情:")
        for error in stats['errors']:
            print(f"  - {error['file']}: {error['error']}")


if __name__ == '__main__':
    main()

