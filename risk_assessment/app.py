import warnings
warnings.filterwarnings("ignore")
import os
import json
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.core.display import HTML
import base64
from io import BytesIO
import mpld3
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import transforms, models
import torch.nn.functional as F

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


class ImagePredictionManager:
    """图片预测模型管理器"""
    def __init__(self):
        self.model_dict = {}
        self.router_model = None
        self.router_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label2idx = {'breast_cancer': 0, 'chest_cancer': 1, 'eye_disease': 2, 'skin_cancer': 3}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.load_all_image_models()
        self.load_router_model()
    
    def load_all_image_models(self):
        """加载所有图片预测模型"""
        try:
            # 加载眼底疾病模型
            eye_disease_models = {
                "DN": {
                    "model": MultiModalPredictor.load("risk_assessment/models/eye_disease/DN"),
                    "label": "脉络膜小疣"
                },
                "DR": {
                    "model": MultiModalPredictor.load("risk_assessment/models/eye_disease/DR"),
                    "label": "糖网病"
                },
                "MH": {
                    "model": MultiModalPredictor.load("risk_assessment/models/eye_disease/MH"),
                    "label": "屈光介质混浊"
                },
                "Normal": {
                    "model": MultiModalPredictor.load("risk_assessment/models/eye_disease/Normal"),
                    "label": "正常"
                },
                "ODC": {
                    "model": MultiModalPredictor.load("risk_assessment/models/eye_disease/ODC"),
                    "label": "视神经盘凹陷"
                },
                "TSLN": {
                    "model": MultiModalPredictor.load("risk_assessment/models/eye_disease/TSLN"),
                    "label": "豹纹状病变"
                }
            }
            self.model_dict["eye_disease"] = eye_disease_models
            print("眼底疾病模型加载完成")
            
            # 加载皮肤癌模型
            self.model_dict["skin_cancer"] = {
                "model": MultiModalPredictor.load("risk_assessment/models/Skin_Cancer"),
                "label": "皮肤癌"
            }
            print("皮肤癌模型加载完成")
            
            # 加载胸部肿瘤模型
            self.model_dict["chest_cancer"] = {
                "model": MultiModalPredictor.load("risk_assessment/models/chest_cancer_detection"),
                "label": "胸部肿瘤"
            }
            print("胸部肿瘤模型加载完成")
            
            # 加载乳腺癌模型
            self.model_dict["breast_cancer"] = {
                "model": YOLO("risk_assessment/models/breast_cancer_detection/breast_cancer_detection/weights/best.pt"),
                "label": "乳腺癌"
            }
            print("乳腺癌模型加载完成")
            
        except Exception as e:
            print(f"加载图片预测模型时出错: {str(e)}")
            # 如果模型文件不存在，创建空的模型字典
            self.model_dict = {}
    
    def load_router_model(self):
        """加载路由模型"""
        try:
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.label2idx))
            model.load_state_dict(torch.load("risk_assessment/models/router_resnet18.pth", map_location=self.router_device))
            model = model.to(self.router_device)
            model.eval()
            self.router_model = model
            print("路由模型加载完成")
        except Exception as e:
            print(f"加载路由模型时出错: {str(e)}")
            self.router_model = None
    
    def predict_image_type(self, input_data, threshold=0.9):
        """
        预测图片类型（路由判断）
        
        Args:
            input_data: 图片路径或PIL.Image对象
            threshold: 置信度阈值，默认0.9
        
        Returns:
            图片类型字符串 ('breast_cancer', 'chest_cancer', 'eye_disease', 'skin_cancer', 'Other_PICTURE')
        """
        if self.router_model is None:
            return "Other_PICTURE"
        
        try:
            if isinstance(input_data, str):
                image = Image.open(input_data).convert("RGB")
            elif isinstance(input_data, Image.Image):
                image = input_data.convert("RGB")
            else:
                raise ValueError("Input must be a file path or PIL.Image")
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.router_device)
            
            with torch.no_grad():
                outputs = self.router_model(image_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy().flatten()
                max_prob = probs.max()
                pred_idx = probs.argmax()
                if max_prob < threshold:
                    return "Other_PICTURE"
                return self.idx2label[pred_idx]
        except Exception as e:
            print(f"路由模型预测出错: {str(e)}")
            return "Other_PICTURE"
    
    def predict_image(self, image_path, img_type):
        """
        预测图片类型
        
        Args:
            image_path: 图片路径
            img_type: 图片类型 ('eye_disease', 'skin_cancer', 'chest_cancer', 'breast_cancer')
        
        Returns:
            预测结果字典
        """
        if img_type == "eye_disease":
            info = "眼底疾病风险评估结果："
            # 先预测是否正常
            if "eye_disease" not in self.model_dict or "Normal" not in self.model_dict["eye_disease"]:
                return {"info": "模型未加载", "label": "Error"}
            
            predictions = self.model_dict['eye_disease']['Normal']['model'].predict({'image': [image_path]})[0]
            proba = self.model_dict['eye_disease']['Normal']['model'].predict_proba({'image': [image_path]})[0]
            if predictions == 1:
                return {"info": info, "label": "Healthy"}
            else:
                for key, value in self.model_dict['eye_disease'].items():
                    if key != "Normal":
                        predictions = value['model'].predict({'image': [image_path]})[0]
                        proba = value['model'].predict_proba({'image': [image_path]})[0]
                        if predictions == 1:
                            info += value['label'] + ",概率：" + str(round(proba[predictions] * 100, 2)) + "%" + "，"
                if info.endswith("，"):
                    info = info[:-1]
                return {"info": info, "label": "No Healthy"}
        
        elif img_type == "skin_cancer":
            info = "皮肤癌风险评估结果："
            if "skin_cancer" not in self.model_dict:
                return {"info": "模型未加载", "label": "Error"}
            
            predictions = self.model_dict['skin_cancer']['model'].predict({'image': [image_path]})[0]
            proba = self.model_dict['skin_cancer']['model'].predict_proba({'image': [image_path]})[0]
            if predictions == 0:
                info += "光化性角化病,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 1:
                info += "基底细胞癌,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 2:
                info += "良性角化病样病变,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 3:
                info += "皮肤纤维瘤,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 4:
                info += "黑色素瘤,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 5:
                info += "黑色素细胞痣,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 6:
                info += "血管病变,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            return {"info": info, "label": "No Healthy"}
        
        elif img_type == "chest_cancer":
            info = "胸部肿瘤风险评估结果："
            if "chest_cancer" not in self.model_dict:
                return {"info": "模型未加载", "label": "Error"}
            
            predictions = self.model_dict['chest_cancer']['model'].predict({'image': [image_path]})[0]
            proba = self.model_dict['chest_cancer']['model'].predict_proba({'image': [image_path]})[0]
            if predictions == 0:
                return {"info": info, "label": "Healthy"}
            elif predictions == 1:
                info += "腺癌,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 2:
                info += "大细胞癌,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            elif predictions == 3:
                info += "鳞状细胞癌,概率：" + str(round(proba[predictions] * 100, 2)) + "%"
            return {"info": info, "label": "No Healthy"}
        
        elif img_type == "breast_cancer":
            info = "乳腺癌风险评估结果："
            if "breast_cancer" not in self.model_dict:
                return {"info": "模型未加载", "label": "Error"}
            
            results = self.model_dict['breast_cancer']['model'].predict(image_path)[0]
            results_cls = results.boxes.cls.to("cpu").tolist()
            results_conf = results.boxes.conf.to("cpu").tolist()
            for i in range(len(results_cls)):
                if results_cls[i] == 1:
                    info += "良性肿瘤,概率：" + str(round(results_conf[i] * 100, 2)) + "%"
                elif results_cls[i] == 2:
                    info += "恶性肿瘤,概率：" + str(round(results_conf[i] * 100, 2)) + "%"
                elif results_cls[i] == 0:
                    info += "健康,概率：" + str(round(results_conf[i] * 100, 2)) + "%"
            im_bgr = results.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            
            return {"results": results, "info": info, "label": "Analyzed", "image": im_rgb}
        
        return {"info": "", "label": "Other"}
    
    def pil_to_base64(self, image):
        """将PIL图像转换为base64字符串"""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


# 初始化图片预测模型管理器
image_prediction_manager = ImagePredictionManager()

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

@app.route('/predict-image-type', methods=['POST'])
def predict_image_type():
    """预测图片类型（路由判断）"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '请提供图片文件'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 保存临时文件
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # 获取阈值参数
            threshold = float(request.form.get('threshold', 0.9))
            
            # 预测图片类型
            img_type = image_prediction_manager.predict_image_type(tmp_path, threshold)
            
            return jsonify({
                'image_type': img_type
            })
        finally:
            # 删除临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return jsonify({
            'error': f'预测图片类型时发生错误: {str(e)}'
        }), 500

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """图片预测"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '请提供图片文件'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 获取图片类型参数
        img_type = request.form.get('image_type')
        if not img_type:
            return jsonify({'error': '请提供图片类型参数'}), 400
        
        # 保存临时文件
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # 进行图片预测
            result = image_prediction_manager.predict_image(tmp_path, img_type)
            
            # 如果结果中包含图片，转换为base64
            if 'image' in result:
                result['image_base64'] = image_prediction_manager.pil_to_base64(result['image'])
                # 移除PIL Image对象，因为无法序列化为JSON
                del result['image']
                if 'results' in result:
                    # 移除YOLO结果对象，因为无法序列化
                    del result['results']
            
            return jsonify(result)
        finally:
            # 删除临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return jsonify({
            'error': f'图片预测时发生错误: {str(e)}'
        }), 500

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