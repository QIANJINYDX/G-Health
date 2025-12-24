import os
import re

# 设置magic-pdf.json路径（绝对路径）
os.environ["MINERU_TOOLS_CONFIG_JSON"] = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/MedicalExaminationAgent/PhysicalExaminationAgent/client/mineru-models/mineru.json"
os.environ["MINERU_MODEL_SOURCE"]="local"
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


def detect_pdf_content(pdf_file_name):
    """
    检测PDF文件内容
    """
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


def detect_image_content(image_file_name):
    """
    检测图片文件内容
    """
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


def detect_office_content(office_file_name):
    """
    检测Office文件内容
    """
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


if __name__ == "__main__":
    print(detect_office_content("/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/MedicalExaminationAgent/PhysicalExaminationAgent/client/app/uploads/1/2023.08.11-23080746114-.pdf"))