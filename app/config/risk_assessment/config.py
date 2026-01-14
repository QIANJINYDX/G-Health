import os
import json

# 模型编号必须与提示词（app/util/agent_config.py）保持一致，
# 以确保 analyze_dialogue() 输出的编号能触发正确的风险评估模型。
risk_types = {
    0: "糖尿病",
    1: "糖尿病（早期）",
    2: "体检糖尿病",
    3: "肥胖",
    4: "心脏病",
    5: "心力衰竭",
    6: "高血压风险",
    7: "肺癌",
    8: "孕产妇健康",
    9: "睡眠障碍",
    10: "中风风险",
    11: "慢性肾病",
    12: "甲状腺癌",
    13: "鼻咽癌",
    14: "卡路里测量",
    15: "心脏风险",
    # 额外模型（提示词未覆盖）：保留但放到后面，避免干扰 0-15 的对齐
    16: "脱发",
    17: "肝炎",
}
class RiskAssessmentConfig:
    def __init__(self):
        self.config_dir = "risk_assessment/data/config"
        self.models = {
            # 0-15：与提示词编号严格对齐
            0: self._load_config('diabetes.json'),
            1: self._load_config('early_diabetes.json'),
            2: self._load_config('physical_examination_diabetes.json'),
            3: self._load_config('obesity.json'),
            4: self._load_config('heart_disease.json'),
            5: self._load_config('heart_failure_clinical_records.json'),
            6: self._load_config('hypertension_risk.json'),
            7: self._load_config('lung_cancer.json'),
            8: self._load_config('maternal_health.json'),
            9: self._load_config('sleep_disorder.json'),
            10: self._load_config('stroke_risk.json'),
            11: self._load_config('kidney_disease.json'),
            12: self._load_config('thyroid_diff.json'),
            13: self._load_config('nasopharyngeal_cancer.json'),
            14: self._load_config('calories.json'),
            15: self._load_config('heart_risk.json'),
            # 16+：额外模型
            16: self._load_config('hair_fall.json'),
            17: self._load_config('hepatitis.json'),
        }
        self.zh_to_en = {
            "糖尿病": "diabetes",
            "肥胖": "obesity",
            "脱发": "hair_fall",
            "心脏病": "heart_disease",
            "心力衰竭": "heart_failure",
            "心脏风险": "heart_risk",
            "肝炎": "hepatitis",
            "高血压风险": "hypertension_risk",
            "肺癌": "lung_cancer",
            "孕产妇健康": "maternal_health",
            "体检糖尿病": "physical_examination_diabetes",
            "睡眠障碍": "sleep_disorder",
            "中风风险": "stroke_risk",
            "卡路里测量": "calories",
            "糖尿病（早期）": "early_diabetes"
            ,"鼻咽癌": "nasopharyngeal_cancer"
        }
        self.en_to_zh = {
            "diabetes": "糖尿病",
            "obesity": "肥胖",
            "hair_fall": "脱发",
            "heart_disease": "心脏病",
            "heart_failure": "心力衰竭",
            "heart_risk": "心脏风险",
            "hepatitis": "肝炎",
            "hypertension_risk": "高血压风险",
            "lung_cancer": "肺癌",
            "maternal_health": "孕产妇健康",
            "physical_examination_diabetes": "体检糖尿病",
            "sleep_disorder": "睡眠障碍",
            "stroke_risk": "中风风险",
            "calories": "卡路里测量",
            "early_diabetes": "糖尿病（早期）"
            ,"nasopharyngeal_cancer": "鼻咽癌"
        }
        
    def _load_config(self, filename):
        """加载特定疾病的配置文件"""
        try:
            with open(os.path.join(self.config_dir, filename), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
            
    def get_model_config(self, model_id):
        """获取特定模型的配置"""
        return self.models.get(model_id)
        
    def get_form_fields(self, model_id, language='zh'):
        """获取特定模型的表单字段
        
        Args:
            model_id: 模型ID
            language: 语言代码，'zh' 或 'en'，默认为 'zh'
        """
        config = self.models.get(model_id)
        if not config:
            return None
            
        fields = []
        for field_name, field_info in config['feature_columns'].items():
            if field_name != config['target_column']:  # 排除目标列
                # 根据语言选择描述和值
                if language == 'en':
                    description = field_info.get('description_en', field_info.get('description', ''))
                    values = field_info.get('values_en', field_info.get('values', []))
                else:
                    description = field_info.get('description', '')
                    values = field_info.get('values', [])
                
                field = {
                    'name': field_name,
                    'description': description,
                    'type': field_info['type'],
                    'values': values
                }
                if 'mapping' in field_info:
                    field['mapping'] = field_info['mapping']
                fields.append(field)
        return fields
        
    def get_model_info(self, model_id, language='zh'):
        """获取模型基本信息
        
        Args:
            model_id: 模型ID
            language: 语言代码，'zh' 或 'en'，默认为 'zh'
        """
        config = self.models.get(model_id)
        if not config:
            return None
        
        # 根据语言选择模型名称
        if language == 'en':
            model_name_display = config.get('model_name_en', config['model_name'])
            # 兼容：model_id=13 在提示词中代表“鼻咽癌”
            if model_id == 13 and model_name_display == config.get('model_name_en', config['model_name']):
                model_name_display = "Nasopharyngeal Cancer"
        else:
            model_name_display = self.en_to_zh.get(config['model_name'], config['model_name'])
            # 兼容：model_id=13 在提示词中代表“鼻咽癌”
            if model_id == 13:
                model_name_display = "鼻咽癌"
        
        result = {
                'model_name': config['model_name'],
            'model_name_display': model_name_display,
            'model_name_zh': self.en_to_zh.get(config['model_name'], config['model_name']),
                'model_type': config['model_type'],
                'model_path': config['model_path']
            }
        
        # 根据语言选择目标映射
        if 'target_mapping_draw' in config:
            if language == 'en' and 'target_mapping_draw_en' in config:
                result['target_mapping'] = config['target_mapping_draw_en']
            else:
                result['target_mapping'] = config['target_mapping_draw']
        
        return result

# 创建全局实例
risk_config = RiskAssessmentConfig() 