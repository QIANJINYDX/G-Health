import json
import os

import requests
from modelscope import snapshot_download


def download_json(url):
    # 下载JSON文件
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.1.1':
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # 定义 magic-pdf-models 目录路径（相对于当前脚本位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    magic_pdf_models_dir = os.path.join(script_dir, 'magic-pdf-models')
    
    # 确保目录存在
    os.makedirs(magic_pdf_models_dir, exist_ok=True)
    
    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small_2501/*",
        "models/MFR/unimernet_small_2503/*"
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    # 将模型下载到 magic-pdf-models 目录
    model_dir = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', 
                                  allow_patterns=mineru_patterns,
                                  cache_dir=magic_pdf_models_dir)
    layoutreader_model_dir = snapshot_download('ppaanngggg/layoutreader',
                                                cache_dir=magic_pdf_models_dir)
    model_dir = model_dir + '/models'
    print(f'model_dir is: {model_dir}')
    print(f'layoutreader_model_dir is: {layoutreader_model_dir}')

    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/magic-pdf.template.json'
    config_file_name = 'magic-pdf.json'
    # 将配置文件保存到 magic-pdf-models 目录
    config_file = os.path.join(magic_pdf_models_dir, config_file_name)

    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f'The configuration file has been configured successfully, the path is: {config_file}')
