import warnings
warnings.filterwarnings("ignore")
import os
import json
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.core.display import HTML
import base64
from io import BytesIO
import mpld3

# 设置 matplotlib 字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

app = Flask(__name__)
# 设置JSON响应的编码配置
# flask 2.3.0以上
app.json.ensure_ascii = False # 解决中文乱码问题
# flask 2.3.0以下
# app.config['JSON_AS_ASCII'] = False
CORS(app)
class AutogluonWrapper:
    def __init__(self, predictor, feature_names,model_type):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.model_type = model_type
    
    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        if self.model_type == 'binary':
            return self.ag_model.predict_proba(X, as_multiclass=False)
        elif self.model_type == 'multiclass':
            return self.ag_model.predict_proba(X)
        elif self.model_type == 'regression':
            return self.ag_model.predict(X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
class ModelManager:
    def __init__(self):
        self.models = {}
        self.configs = {}
        self.feature_columns = {}
        self.datasets = {}  # New dictionary to store datasets
        self.dataset_medians = {}  # New dictionary to store dataset medians
        self.wrappers = {}  # New dictionary to store AutogluonWrapper instances
        self.explainers = {}  # New dictionary to store SHAP explainers
        self.load_all_models()
    
    def load_config(self, config_file):
        """加载单个配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_model(self, config):
        """加载单个模型"""
        model_path = config['model_path']
        return TabularPredictor.load(model_path,require_version_match=False,require_py_version_match=False)
    
    def load_dataset(self, config):
        """加载数据集"""
        data_path = config['data_path']
        return pd.read_csv(data_path)
    
    def calculate_medians(self, dataset, config):
        """计算数据集的median值"""
        # 获取特征列
        feature_cols = list(config['feature_columns'].keys())
        feature_cols.remove(config['target_column'])
        
        # 计算median
        medians = dataset[feature_cols].median()
        return medians
    
    def load_all_models(self):
        """加载所有模型和配置"""
        config_dir = os.path.join('risk_assessment', 'data', 'config')
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"配置目录不存在: {config_dir}")
        
        # 遍历配置目录中的所有JSON文件
        for config_file in os.listdir(config_dir):
            if config_file.endswith('.json'):
                model_name = os.path.splitext(config_file)[0]
                config_path = os.path.join(config_dir, config_file)
                
                try:
                    print(f"加载模型: {model_name}",end=" ")
                    # 加载配置
                    print(f"加载配置",end=" ")
                    config = self.load_config(config_path)
                    
                    # 加载模型
                    print(f"加载模型",end=" ")
                    predictor = self.load_model(config)
                    
                    # 加载数据集
                    print(f"加载数据集",end=" ")
                    dataset = self.load_dataset(config)
                    
                    # 计算median
                    print(f"计算median",end=" ")
                    medians = self.calculate_medians(dataset, config)
                    
                    # 获取特征列
                    print(f"获取特征列",end=" ")
                    feature_cols = list(config['feature_columns'].keys())
                    feature_cols.remove(config['target_column'])
                    
                    # 创建AutogluonWrapper
                    print(f"创建AutogluonWrapper",end=" ")
                    wrapper = AutogluonWrapper(predictor, feature_cols,config['model_type'])
                    
                    # 创建SHAP解释器
                    print(f"创建SHAP解释器",end=" ")
                    explainer = shap.KernelExplainer(wrapper.predict, medians)
                    
                    # 存储模型相关信息
                    print(f"存储模型相关信息",end=" ")
                    self.models[model_name] = predictor
                    self.configs[model_name] = config
                    self.feature_columns[model_name] = feature_cols
                    self.datasets[model_name] = dataset
                    self.dataset_medians[model_name] = medians
                    self.wrappers[model_name] = wrapper
                    self.explainers[model_name] = explainer
                    
                    print(f"成功加载模型: {model_name}")
                except Exception as e:
                    print(f"加载模型 {model_name} 时出错: {str(e)}")
    
    def get_available_models(self):
        """获取所有可用模型列表"""
        return list(self.models.keys())
    
    def map_feature_value(self, feature_name, value, feature_config):
        """根据特征配置中的mapping规则转换特征值"""
        # 如果特征配置中没有mapping，返回原值
        if 'mapping' not in feature_config:
            return value
            
        mapping = feature_config['mapping']
        
        # 如果值已经是数值类型且不在映射中，直接返回
        if isinstance(value, (int, float)) and str(value) not in mapping.values():
            return value
            
        # 将值转换为字符串进行映射
        str_value = str(value)
        
        # 如果值在mapping中，返回映射值
        if str_value in mapping:
            return mapping[str_value]
            
        # 如果存在英文取值列表，尝试按索引映射到中文列表再取mapping
        values_en = feature_config.get('values_en')
        values = feature_config.get('values')
        if values_en and values:
            try:
                idx = values_en.index(str_value)
                zh_value = values[idx]
                if zh_value in mapping:
                    return mapping[zh_value]
            except ValueError:
                pass
            
        # 如果值已经是映射后的值，直接返回
        if str_value in [str(v) for v in mapping.values()]:
            return float(str_value) if '.' in str_value else int(str_value)
            
        # 如果找不到映射，返回原值
        return value
    
    def preprocess_input_data(self, input_data, config):
        """预处理输入数据，应用特征映射"""
        feature_configs = config.get('feature_columns', {})
        mapped_data = input_data.copy()
        
        for feature in input_data.columns:
            if feature in feature_configs:
                mapped_data[feature] = input_data[feature].apply(
                    lambda x: self.map_feature_value(feature, x, feature_configs[feature])
                )
        
        print("输入数据:", input_data)
        print("映射数据:", mapped_data)
        return mapped_data
    
    def predict(self, model_name, input_data):
        """使用指定模型进行预测"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        predictor = self.models[model_name]
        config = self.configs[model_name]
        # 只使用既在配置中又在输入数据中的特征列，且排除目标列
        all_feature_columns = list(config.get('feature_columns', {}).keys())
        target_col = config.get('target_column')
        feature_columns = [c for c in all_feature_columns if c != target_col and c in input_data.columns]
        print(f"调用模型: {model_name}")
        print(f"模型配置: {config}")
        print("--------------------------------输入处理--------------------------------")
        processed_data = self.preprocess_input_data(input_data, config)

        # 仅保留训练使用的特征列，避免多余列导致形状不一致
        processed_data = processed_data[feature_columns]
        input_data_display = input_data[feature_columns] if isinstance(input_data, pd.DataFrame) else input_data
        
        print("--------------------------------预测--------------------------------")
        if config['model_type'] == 'regression':
            prediction = predictor.predict(processed_data)
            probabilities = None
        else:
            prediction = predictor.predict(processed_data)
            probabilities = predictor.predict_proba(processed_data)
        print("--------------------------------预测完成--------------------------------")

        print(f"prediction: {prediction}")
        print(f"probabilities: {probabilities}")
        
        # 获取分类标签
        if 'target_mapping_draw' in config:
            class_labels = [config['target_mapping_draw'][str(i)] for i in range(len(config['target_mapping_draw']))]
        else:
            class_labels = predictor.class_labels
            
        # 准备SHAP可视化数据
        X = processed_data.copy()  # 使用处理后的数据
        y = prediction
        X_display = input_data_display
        y_display = prediction
        shap_plot = self.visualize_shap(model_name, X, y, X_display, y_display, feature_columns)
        print(f"shap_plot: {shap_plot}")
        
        return prediction, probabilities, class_labels, shap_plot
    
    def visualize_shap(self, model_name, X, y, X_display, y_display, feature_columns):
        """可视化SHAP值"""
        try:
            # 确保X是DataFrame格式
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # 仅保留模型训练的特征列，确保形状一致
            X = X[feature_columns]
            
            # 使用背景数据（中位数）作为参考点
            background = self.dataset_medians[model_name][feature_columns].to_frame().T
            if not self.configs[model_name]['model_type'] == 'regression':
                # 对于多分类问题，我们需要为每个类别计算SHAP值
                shap_values = []
                for i in range(len(self.configs[model_name]['target_mapping_draw'])):
                    print(f"计算类别 {i} 的SHAP值...")
                    # 使用当前类别的概率作为目标
                    def f(x):
                        if not isinstance(x, pd.DataFrame):
                            x = pd.DataFrame(x, columns=feature_columns)
                        proba = self.models[model_name].predict_proba(x)
                        return proba.iloc[:, i] if isinstance(proba, pd.DataFrame) else proba[:, i]
                    
                    # 计算SHAP值
                    explainer = shap.KernelExplainer(f, background)
                    shap_values.append(explainer.shap_values(X))
                    print(f"类别 {i} 的SHAP值计算完成")
                
                # 选择预测的类别
                predicted_class = int(y.iloc[0])
                print(f"预测的类别: {predicted_class}")
                
                # 创建force plot
                plt.figure(figsize=(10, 4))
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[predicted_class][0],
                    X.iloc[0],
                    show=False,
                    matplotlib=True
                )
                plt.show()
                sio = BytesIO()
                plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
                data = base64.encodebytes(sio.getvalue()).decode()
                shap_html = 'data:image/png;base64,' + str(data)
                
                return shap_html
            else:
                shap_values = self.explainers[model_name].shap_values(X)
                # print(f"shap_values: {shap_values}")
                plt.figure(figsize=(10, 4))
                shap.force_plot(
                    self.explainers[model_name].expected_value,
                    shap_values,
                    X.iloc[0],
                    show=False,
                    matplotlib=True
                )
                plt.show()
                sio = BytesIO()
                plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
                data = base64.encodebytes(sio.getvalue()).decode()
                shap_html = 'data:image/png;base64,' + str(data)
                return shap_html

        except Exception as e:
            print(f"SHAP可视化过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# 初始化模型管理器
model_manager = ModelManager()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'message': '风险评估服务正常运行',
        'available_models': model_manager.get_available_models()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """列出所有可用的模型"""
    model_mapping = {
        "breast_cancer": "乳腺癌",
        "calories": "热量",
        "cervical_cancer": "宫颈癌",
        "colorectal_cancer": "结直肠癌",
        "diabetes": "糖尿病",
        "early_diabetes": "早期糖尿病",
        "hair_fall": "脱发",
        "heart_disease": "心脏病",
        "heart_failure_clinical_records": "心力衰竭临床记录",
        "heart_risk": "心脏病风险",
        "hepatitis": "肝炎",
        "hypertension_risk": "高血压风险",
        "kidney_disease": "肾病",
        "lung_cancer": "肺癌",
        "maternal_health": "孕产妇健康",
        "nasopharyngeal_cancer": "鼻咽癌",
        "obesity": "肥胖",
        "pancreatic_cancer": "胰腺癌",
        "physical_examination_diabetes": "体检糖尿病",
        "prostate_cancer": "前列腺癌",
        "sleep_disorder": "睡眠障碍",
        "stroke_risk": "中风风险",
        "thyroid_diff": "甲状腺结节分级"
    }


    models_info = {}
    for model_name in model_manager.get_available_models():
        config = model_manager.configs[model_name]
        models_info[model_name] = {
            'feature_columns': model_manager.feature_columns[model_name],
            'target_column': config['target_column'],
            'model_type': config['model_type'],
            'model_name': model_mapping[model_name]
        }
    return jsonify(models_info)

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """使用指定模型进行预测"""
    try:
        # 检查模型是否存在
        if model_name not in model_manager.models:
            return jsonify({
                'error': f'模型 {model_name} 不存在'
            }), 404
        
        # 获取输入数据
        data = request.get_json()
        # 读取语言参数，默认中文
        language = data.pop('language', 'zh') if isinstance(data, dict) else 'zh'
        
        # 验证输入数据
        if not data or not isinstance(data, dict):
            return jsonify({
                'error': '无效的输入数据格式，请提供JSON对象'
            }), 400
        
        # 检查必需的特征
        feature_columns = model_manager.feature_columns[model_name]
        missing_features = [f for f in feature_columns if f not in data]
        if missing_features:
            return jsonify({
                'error': f'缺少必需的特征: {", ".join(missing_features)}'
            }), 400
        
        # 创建输入DataFrame
        input_data = pd.DataFrame([data])
        
        # 进行预测
        prediction, probabilities, class_labels, shap_plot = model_manager.predict(model_name, input_data)
        print("--------------------------------预测完成--------------------------------")
        print(f"prediction: {prediction}")
        print(f"probabilities: {probabilities}")
        print(f"class_labels: {class_labels}")
        # print(f"shap_plot: {shap_plot}")
        if model_manager.configs[model_name]['model_type'] == 'regression':
            result = {
                'prediction': float(prediction.iloc[0]),
                "shap_plot": shap_plot
            }
        else:
            # 根据语言选择标签映射
            mapping_draw = model_manager.configs[model_name].get(
                'target_mapping_draw_en' if language == 'en' else 'target_mapping_draw',
                model_manager.configs[model_name].get('target_mapping_draw', {})
            )
            labels = [mapping_draw.get(str(i), str(i)) for i in range(len(mapping_draw))]
            result = {
                'prediction': int(prediction.iloc[0]),
                'prediction_label': labels[int(prediction.iloc[0])] if labels else class_labels[int(prediction.iloc[0])],
                'probabilities': {
                    (labels[i] if labels else class_labels[i]): float(probabilities.iloc[0, i])
                    for i in range(len(class_labels))
                },
                "shap_plot": shap_plot
            }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'预测过程中发生错误: {str(e)}'
        }), 500

@app.route('/model-info/<model_name>', methods=['GET'])
def model_info(model_name):
    """获取指定模型的信息"""
    if model_name not in model_manager.models:
        return jsonify({
            'error': f'模型 {model_name} 不存在'
        }), 404
    
    config = model_manager.configs[model_name]
    return jsonify({
        'feature_columns': model_manager.feature_columns[model_name],
        'target_column': config['target_column'],
        'model_type': config['model_type'],
        'model_name': config['model_name']
    })

if __name__ == '__main__':
    mode='development'
    mode='production'
    if mode == 'development':
        app.run(debug=True,use_reloader=False)
    else:
        from werkzeug.serving import run_simple
        run_simple('0.0.0.0', 5002, app)
    # finally:
    #     print("risk_assessment_process.terminate()")
    #     app.terminate()
    #     app.wait()