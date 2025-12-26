"""
文件检测模块 - 支持本地解析和API解析两种方式

使用说明：
1. 默认使用本地解析方式
2. 要使用API解析方式，需要设置以下环境变量：
   - FILE_PARSE_MODE="api"  # 设置为API模式
   - MINERU_API_TOKEN="your_token_here"  # 从官网申请的API token
   - MINERU_API_MODEL_VERSION="vlm"  # 模型版本（可选，默认为vlm）
   - MINERU_FILE_UPLOAD_URL="https://your-file-upload-service.com/upload"  # 文件上传服务URL（必需）

3. API解析方式需要先将文件上传到可访问的URL，然后调用解析API
   如果配置了MINERU_FILE_UPLOAD_URL，系统会自动上传文件并获取URL
   如果未配置，需要手动处理文件上传

示例环境变量配置：
export FILE_PARSE_MODE="api"
export MINERU_API_TOKEN="your_api_token"
export MINERU_FILE_UPLOAD_URL="https://your-file-upload-service.com/upload"
"""

import os
import re
import requests
import base64
from typing import Optional

# ========== 全局配置开关 ==========
# 解析方式开关: "local" 使用本地解析, "api" 使用API解析
PARSE_MODE = os.getenv("FILE_PARSE_MODE", "local")  # 默认为本地解析

# API配置
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")  # 从环境变量读取token，如果没有则使用空字符串
MINERU_API_URL = "https://mineru.net/api/v4/extract/task"
MINERU_API_MODEL_VERSION = os.getenv("MINERU_API_MODEL_VERSION", "vlm")  # 模型版本，默认为vlm
# 文件上传服务URL（如果需要先上传文件到可访问的URL，可以配置此项）
# 如果为空，API解析将无法工作（因为API需要文件URL）
MINERU_FILE_UPLOAD_URL = os.getenv("MINERU_FILE_UPLOAD_URL", "")

# ========== 本地解析相关导入（仅在本地模式时导入） ==========
if PARSE_MODE == "local":
    # 设置magic-pdf.json路径（绝对路径）
    # os.environ["MINERU_TOOLS_CONFIG_JSON"] = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/MedicalExaminationAgent/PhysicalExaminationAgent/client/mineru-models/mineru.json"
    os.environ["MINERU_MODEL_SOURCE"] = "local"
    from mineru.cli.common import read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

    local_image_dir, local_md_dir = "output/images", "output/md"
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    image_dir = str(os.path.basename(local_image_dir))


def clean_html_table_tags(content):
    """
    清洗HTML表格标签和其他HTML标签，保留文本内容
    
    Args:
        content: 包含HTML标签的字符串内容
        
    Returns:
        清洗后的纯文本内容
    """
    if not content:
        return content
    
    # 移除所有HTML表格标签（table, tr, td, th, tbody, thead, tfoot等）
    # 包括带属性的标签，如 <td rowspan=5 colspan=1>
    content = re.sub(r'<table[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</table>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<tr[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</tr>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<td[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</td>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<th[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</th>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<tbody[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</tbody>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<thead[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</thead>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<tfoot[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</tfoot>', '', content, flags=re.IGNORECASE)
    
    # 移除其他常见的HTML标签（可选，根据需要调整）
    # content = re.sub(r'<[^>]+>', '', content)  # 移除所有HTML标签
    
    # 清理多余的空白字符
    # 将多个连续的空格、制表符、换行符压缩为单个空格
    content = re.sub(r'\s+', ' ', content)
    
    # 清理行首行尾的空白
    content = content.strip()
    
    return content


def detect_content_via_api(file_path: str, file_type: str = "pdf") -> Optional[str]:
    """
    通过API方式解析文件内容
    
    根据用户提供的示例，API使用URL方式。对于本地文件，有两种处理方式：
    1. 如果配置了文件上传服务URL，先上传文件获取URL，然后调用解析API
    2. 如果未配置上传服务，尝试使用multipart/form-data直接上传文件
    
    Args:
        file_path: 文件路径
        file_type: 文件类型 ("pdf", "image", "office")
        
    Returns:
        解析后的Markdown内容，如果失败返回None
    """
    if not MINERU_API_TOKEN:
        print("Warning: MINERU_API_TOKEN is not set, cannot use API mode")
        return None
    
    try:
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MINERU_API_TOKEN}"
        }
        
        # 获取文件URL
        file_url = None
        
        # 方式1: 如果配置了文件上传服务，先上传文件获取URL
        if MINERU_FILE_UPLOAD_URL:
            file_url = _upload_file_to_url(file_path)
            if not file_url:
                print("文件上传失败，无法获取文件URL")
                return None
        else:
            # 方式2: 尝试直接使用multipart/form-data上传文件
            # 如果API支持直接上传，使用这种方式
            # 否则需要配置MINERU_FILE_UPLOAD_URL
            print("Warning: MINERU_FILE_UPLOAD_URL not set, trying direct file upload...")
            # 这里可以尝试直接上传，但根据示例，API可能需要URL
            # 如果API支持直接上传，可以在这里实现
            return None
        
        # 准备请求数据（按照用户提供的示例格式）
        data = {
            "url": file_url,  # 文件的可访问URL
            "model_version": MINERU_API_MODEL_VERSION
        }
        
        # 发送请求
        response = requests.post(
            MINERU_API_URL,
            headers=headers,
            json=data,
            timeout=300  # 5分钟超时，因为文件解析可能需要较长时间
        )
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result:
                # 提取解析结果
                md_content = result["data"]
                # 清洗HTML表格标签
                md_content = clean_html_table_tags(md_content)
                print(f"API解析成功: {file_path}")
                return md_content
            else:
                print(f"API响应格式错误: {result}")
                return None
        else:
            print(f"API请求失败: status_code={response.status_code}, response={response.text}")
            return None
            
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"API解析出错: {str(e)}")
        return None


def _upload_file_to_url(file_path: str) -> Optional[str]:
    """
    上传文件到文件服务并获取可访问的URL
    
    Args:
        file_path: 本地文件路径
        
    Returns:
        文件的可访问URL，如果失败返回None
    """
    if not MINERU_FILE_UPLOAD_URL:
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            response = requests.post(
                MINERU_FILE_UPLOAD_URL,
                files=files,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # 根据实际的文件上传服务响应格式调整
                # 假设响应格式为 {"url": "https://..."} 或 {"data": {"url": "https://..."}}
                if "url" in result:
                    return result["url"]
                elif "data" in result and "url" in result["data"]:
                    return result["data"]["url"]
                else:
                    print(f"文件上传服务响应格式错误: {result}")
                    return None
            else:
                print(f"文件上传失败: status_code={response.status_code}, response={response.text}")
                return None
    except Exception as e:
        print(f"文件上传出错: {str(e)}")
        return None




def detect_pdf_content(pdf_file_name):
    """
    检测PDF文件内容
    根据全局开关 PARSE_MODE 选择使用本地解析或API解析
    """
    # 根据全局开关选择解析方式
    if PARSE_MODE == "api":
        result = detect_content_via_api(pdf_file_name, "pdf")
        if result:
            return result
        # 如果API解析失败，可以回退到本地解析（可选）
        print("API解析失败，尝试使用本地解析...")
    
    # 本地解析
    if PARSE_MODE == "local":
        try:
            # 读取PDF文件
            pdf_bytes = read_fn(pdf_file_name)
            
            # 使用pipeline后端进行解析
            pdf_bytes_list = [pdf_bytes]
            p_lang_list = ["ch"]  # 默认中文
            parse_method = "auto"  # 自动检测方法
            
            # 执行解析
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                pdf_bytes_list, 
                p_lang_list, 
                parse_method=parse_method, 
                formula_enable=True,
                table_enable=True
            )
            
            # 处理结果
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            # 转换为中间JSON格式
            middle_json = pipeline_result_to_middle_json(
                model_list, 
                images_list, 
                pdf_doc, 
                image_writer, 
                _lang, 
                _ocr_enable
                # formula_enable=True
            )
            
            # 提取PDF信息并生成Markdown
            pdf_info = middle_json["pdf_info"]
            md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            
            # 清洗HTML表格标签
            md_content = clean_html_table_tags(md_content)
            
            print(md_content)
            return md_content
        except Exception as e:
            print(f"本地解析出错: {str(e)}")
            return None
    else:
        # API模式但解析失败
        return None


def detect_image_content(image_file_name):
    """
    检测图片文件内容
    根据全局开关 PARSE_MODE 选择使用本地解析或API解析
    """
    # 根据全局开关选择解析方式
    if PARSE_MODE == "api":
        result = detect_content_via_api(image_file_name, "image")
        if result:
            return result
        # 如果API解析失败，可以回退到本地解析（可选）
        print("API解析失败，尝试使用本地解析...")
    
    # 本地解析
    if PARSE_MODE == "local":
        try:
            # 读取图片文件
            pdf_bytes = read_fn(image_file_name)
            
            # 使用pipeline后端进行解析，图片强制使用OCR模式
            pdf_bytes_list = [pdf_bytes]
            p_lang_list = ["ch"]  # 默认中文
            parse_method = "ocr"  # 图片使用OCR方法
            
            # 执行解析
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                pdf_bytes_list, 
                p_lang_list, 
                parse_method=parse_method, 
                formula_enable=True,
                table_enable=True
            )
            
            # 处理结果
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            # 转换为中间JSON格式
            middle_json = pipeline_result_to_middle_json(
                model_list, 
                images_list, 
                pdf_doc, 
                image_writer, 
                _lang, 
                _ocr_enable
                # formula_enable=True
            )
            
            # 提取PDF信息并生成Markdown
            pdf_info = middle_json["pdf_info"]
            md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            
            # 清洗HTML表格标签
            md_content = clean_html_table_tags(md_content)
            
            print(md_content)
            
            return md_content
        except Exception as e:
            print(f"本地解析出错: {str(e)}")
            return None
    else:
        # API模式但解析失败
        return None


def detect_office_content(office_file_name):
    """
    检测Office文件内容
    根据全局开关 PARSE_MODE 选择使用本地解析或API解析
    """
    # 根据全局开关选择解析方式
    if PARSE_MODE == "api":
        result = detect_content_via_api(office_file_name, "office")
        if result:
            return result
        # 如果API解析失败，可以回退到本地解析（可选）
        print("API解析失败，尝试使用本地解析...")
    
    # 本地解析
    if PARSE_MODE == "local":
        try:
            # 读取Office文件
            pdf_bytes = read_fn(office_file_name)
            
            # 使用pipeline后端进行解析，Office文件使用OCR模式
            pdf_bytes_list = [pdf_bytes]
            p_lang_list = ["ch"]  # 默认中文
            parse_method = "ocr"  # Office文件使用OCR方法
            
            # 执行解析
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                pdf_bytes_list, 
                p_lang_list, 
                parse_method=parse_method, 
                formula_enable=True,
                table_enable=True
            )
            
            # 处理结果
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            # 转换为中间JSON格式
            middle_json = pipeline_result_to_middle_json(
                model_list, 
                images_list, 
                pdf_doc, 
                image_writer, 
                _lang, 
                _ocr_enable
                # formula_enable=True
            )
            
            # 提取PDF信息并生成Markdown
            pdf_info = middle_json["pdf_info"]
            md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            
            # 清洗HTML表格标签
            md_content = clean_html_table_tags(md_content)
            
            print(md_content)
            return md_content
        except Exception as e:
            print(f"本地解析出错: {str(e)}")
            return None
    else:
        # API模式但解析失败
        return None


if __name__ == "__main__":
    print(detect_office_content("/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/MedicalExaminationAgent/PhysicalExaminationAgent/client/app/uploads/1/2023.08.11-23080746114-.pdf"))