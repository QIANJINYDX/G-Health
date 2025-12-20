import warnings
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from autogluon.multimodal.utils.object_detection import from_coco
from ultralytics import YOLO
import torch
import json

DEFAULT_TRAIN_NUM_WORKERS = int(os.getenv("AG_TRAIN_NUM_WORKERS", "0"))
DEFAULT_INFER_NUM_WORKERS = int(os.getenv("AG_INFER_NUM_WORKERS", "0"))
DEFAULT_PER_GPU_BATCH_SIZE = int(os.getenv("AG_PER_GPU_BATCH_SIZE", "16"))
def train_model(
    train_data_path: str,
    model_save_path: str,
    time_limit: int = 30,
    num_workers_train: int = 4,
    num_workers_inference: int = 0,
    per_gpu_batch_size: int | None = None,
):
    """
    Train the model with the given data
    
    Args:
        train_data_path: path to training data CSV
        model_save_path: path to save the trained model
        time_limit: training time limit in seconds
        num_workers_train: training dataloader worker count for higher throughput
        num_workers_inference: eval/inference worker count (keep low to avoid shm issues)
        per_gpu_batch_size: override per-gpu batch size to better saturate GPU
    """
    train_data = pd.read_csv(train_data_path)
    predictor = MultiModalPredictor(label="label", path=model_save_path)

    hyperparameters = {
        "env.num_workers": max(num_workers_train, 0),
        "env.num_workers_inference": max(num_workers_inference, 0),
    }
    if per_gpu_batch_size is not None:
        hyperparameters["env.per_gpu_batch_size"] = per_gpu_batch_size

    predictor.fit(
        train_data=train_data,
        time_limit=time_limit,
        hyperparameters=hyperparameters,
    )
    return predictor

def evaluate_model(test_data_path: str, model_path: str, results_save_path: str):
    """
    Evaluate the model and save results including ROC curve, PR curve, and confusion matrix
    
    Args:
        test_data_path: path to test data CSV
        model_path: path to load the trained model
        results_save_path: path to save evaluation results and plots
    
    Saves:
        - roc_curves.png: ROC curve plot
        - pr_curve.png: Precision-Recall curve plot
        - metrics.txt: Text file with evaluation metrics
        - plot_data.json: Complete plot data (ROC, PR curves, confusion matrix) for later plotting
        - confusion_matrix.csv: Confusion matrix in CSV format
        - roc_curve_data.csv: ROC curve data (fpr, tpr) for binary classification
        - pr_curve_data.csv: PR curve data (precision, recall) for binary classification
        - roc_curve_data_micro_avg.csv: Micro-average ROC curve data for multi-class
        - pr_curve_data_micro_avg.csv: Micro-average PR curve data for multi-class
        - roc_curve_data_class_X.csv: Per-class ROC curve data for multi-class (one file per class)
    """
    # Load test data and model
    test_data = pd.read_csv(test_data_path)
    predictor = MultiModalPredictor.load(model_path)
    
    # Get predictions and probabilities
    probs = predictor.predict_proba(test_data)
    preds = predictor.predict(test_data)
    true_labels = test_data['label'].values
    
    # Get unique classes
    classes = np.unique(true_labels)
    n_classes = len(classes)
    
    # Calculate basic metrics
    accuracy = (preds == true_labels).mean()
    f1 = f1_score(true_labels, preds, average='weighted')
    precision = precision_score(true_labels, preds, average='weighted')
    recall = recall_score(true_labels, preds, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, preds, labels=classes)
    
    # Initialize metrics dictionary
    metrics = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
    }
    
    # Initialize data storage for plotting
    plot_data = {
        'classes': classes.tolist(),
        'n_classes': int(n_classes),
        'roc_data': {},
        'pr_data': {},
        'confusion_matrix': cm.tolist()
    }
    
    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    
    if n_classes == 2:
        # Binary classification case
        fpr, tpr, _ = roc_curve(true_labels, probs.iloc[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        metrics['AUC'] = roc_auc
        
        # Save ROC data
        plot_data['roc_data']['binary'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }
        
        # Calculate PR curve and AUPR for binary case
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, probs.iloc[:, 1])
        aupr = average_precision_score(true_labels, probs.iloc[:, 1])
        
        # Save PR data
        plot_data['pr_data']['binary'] = {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist(),
            'aupr': float(aupr)
        }
        
    else:
        # Multi-class case
        # Convert true labels to one-hot encoding
        true_labels_bin = label_binarize(true_labels, classes=classes)
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plot_data['roc_data']['per_class'] = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], probs.iloc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                    label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})')
            metrics[f'Class {i} AUC'] = roc_auc[i]
            
            # Save per-class ROC data
            plot_data['roc_data']['per_class'][f'class_{i}'] = {
                'fpr': fpr[i].tolist(),
                'tpr': tpr[i].tolist(),
                'auc': float(roc_auc[i])
            }
        
        # Calculate micro-average ROC curve and ROC area
        fpr_micro, tpr_micro, _ = roc_curve(true_labels_bin.ravel(), probs.values.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        plt.plot(fpr_micro, tpr_micro,
                label=f'Micro-average ROC curve (AUC = {roc_auc_micro:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        metrics['Micro-average AUC'] = roc_auc_micro
        
        # Save micro-average ROC data
        plot_data['roc_data']['micro_average'] = {
            'fpr': fpr_micro.tolist(),
            'tpr': tpr_micro.tolist(),
            'auc': float(roc_auc_micro)
        }
        
        # Calculate PR curve and AUPR for multi-class case
        precision_curve, recall_curve, _ = precision_recall_curve(
            true_labels_bin.ravel(), probs.values.ravel())
        aupr = average_precision_score(true_labels_bin, probs, average='micro')
        
        # Save PR data for multi-class
        plot_data['pr_data']['micro_average'] = {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist(),
            'aupr': float(aupr)
        }
    
    # Finish ROC plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.savefig(os.path.join(results_save_path, 'roc_curves.png'),dpi=300,bbox_inches="tight")
    plt.close()
    
    # Plot PR curve
    plt.figure(figsize=(6, 6))
    plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
             label=f'Precision-Recall curve (AUPR = {aupr:.2f})')
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curve', fontsize=16, pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_save_path, 'pr_curve.png'),dpi=300,bbox_inches="tight")
    plt.close()
    
    # Add AUPR to metrics
    metrics['AUPR'] = aupr
    
    # Save metrics to file
    with open(os.path.join(results_save_path, 'metrics.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value:.4f}\n')
    
    # Save plot data as JSON for later plotting
    plot_data_path = os.path.join(results_save_path, 'plot_data.json')
    with open(plot_data_path, 'w', encoding='utf-8') as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    
    # Save confusion matrix as CSV for easy viewing
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_path = os.path.join(results_save_path, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    
    # Save ROC curve data as CSV (for binary or micro-average)
    if n_classes == 2:
        roc_df = pd.DataFrame({
            'fpr': plot_data['roc_data']['binary']['fpr'],
            'tpr': plot_data['roc_data']['binary']['tpr']
        })
        roc_df.to_csv(os.path.join(results_save_path, 'roc_curve_data.csv'), index=False)
        
        pr_df = pd.DataFrame({
            'precision': plot_data['pr_data']['binary']['precision'],
            'recall': plot_data['pr_data']['binary']['recall']
        })
        pr_df.to_csv(os.path.join(results_save_path, 'pr_curve_data.csv'), index=False)
    else:
        # Save micro-average ROC and PR curves
        roc_df = pd.DataFrame({
            'fpr': plot_data['roc_data']['micro_average']['fpr'],
            'tpr': plot_data['roc_data']['micro_average']['tpr']
        })
        roc_df.to_csv(os.path.join(results_save_path, 'roc_curve_data_micro_avg.csv'), index=False)
        
        pr_df = pd.DataFrame({
            'precision': plot_data['pr_data']['micro_average']['precision'],
            'recall': plot_data['pr_data']['micro_average']['recall']
        })
        pr_df.to_csv(os.path.join(results_save_path, 'pr_curve_data_micro_avg.csv'), index=False)
        
        # Save per-class ROC curves
        for class_key, class_data in plot_data['roc_data']['per_class'].items():
            class_roc_df = pd.DataFrame({
                'fpr': class_data['fpr'],
                'tpr': class_data['tpr']
            })
            class_roc_df.to_csv(os.path.join(results_save_path, f'roc_curve_data_{class_key}.csv'), index=False)
    
    print(f"Plot data saved to: {plot_data_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    
    return metrics

# def train_model_DR(train_data_path: str, model_save_path: str, time_limit: int = 30):


def train_model_DR(
    train_data_path: str,
    test_data_path: str,
    model_path: str,
    results_path: str,
    time_limit: int = 600,
    num_workers_train: int = 4,
    num_workers_inference: int = 0,
    per_gpu_batch_size: int | None = None,
):
    os.makedirs(results_path, exist_ok=True)
    
    # Train model
    if not os.path.exists(model_path):
        predictor = train_model(
            train_data_path,
            model_path,
            time_limit=time_limit,
            num_workers_train=num_workers_train,
            num_workers_inference=num_workers_inference,
            per_gpu_batch_size=per_gpu_batch_size,
        )
    else:
        predictor = MultiModalPredictor.load(model_path)
        if hasattr(predictor, "_config") and hasattr(predictor._config, "env"):
            predictor._config.env.num_workers = max(num_workers_train, 0)
            predictor._config.env.num_workers_inference = max(num_workers_inference, 0)
            if (
                per_gpu_batch_size is not None
                and getattr(predictor._config.env, "per_gpu_batch_size", None)
                != per_gpu_batch_size
            ):
                predictor._config.env.per_gpu_batch_size = per_gpu_batch_size
    
    # Evaluate model
    metrics = evaluate_model(test_data_path, model_path, results_path)
    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save metrics as JSON
    metrics_json_path = os.path.join(results_path, 'metrics.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_json_path}")
def evaluate_yolo_model(model, yaml_path: str, test_images_dir: str, test_labels_dir: str, results_path: str):
    """
    在测试集上评估YOLO模型并保存结果
    
    Args:
        model: YOLO模型实例
        yaml_path: 原始数据集yaml配置文件路径
        test_images_dir: 测试集图像目录
        test_labels_dir: 测试集标签目录
        results_path: 结果保存路径
    """
    import yaml
    from pathlib import Path
    
    # 创建临时测试集yaml文件
    test_yaml_path = os.path.join(results_path, 'test_dataset.yaml')
    dataset_base_dir = os.path.dirname(yaml_path)
    
    # 读取原始yaml文件获取类别信息
    original_yaml_path = yaml_path
    with open(original_yaml_path, 'r') as f:
        original_config = yaml.safe_load(f)
    
    # 创建测试集yaml配置（使用绝对路径）
    # YOLO的val方法默认使用val集，所以我们将val路径设置为test路径
    # 这样val方法就会在测试集上评估
    dataset_base_dir_abs = os.path.abspath(dataset_base_dir)
    test_config = {
        'path': dataset_base_dir_abs,
        'train': original_config.get('train', 'images/train'),  # 保留train路径（YOLO要求）
        'val': 'images/test',  # 将val路径设置为test路径，这样val方法会在测试集上评估
        'test': 'images/test',  # 测试集路径（保留用于参考）
        'names': original_config['names']
    }
    
    with open(test_yaml_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, sort_keys=False)
    
    # 在测试集上评估
    print(f"在测试集上评估模型...")
    print(f"测试图像目录: {test_images_dir}")
    print(f"测试标签目录: {test_labels_dir}")
    print(f"测试数据集YAML: {test_yaml_path}")
    print("注意: YOLO的val方法使用val集，已将val路径设置为test路径")
    
    # 创建预测结果保存目录
    predictions_dir = os.path.join(results_path, 'test_predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # 使用val方法评估测试集
    # 由于我们将yaml中的val路径设置为test路径，val方法会在测试集上评估
    try:
        metrics = model.val(
            data=test_yaml_path,
            imgsz=640,
            batch=8,
            conf=0.25,  # 置信度阈值
            iou=0.7,    # IoU阈值
            device='0' if torch.cuda.is_available() else 'cpu',
            workers=0,  # 设置为0避免共享内存不足的问题
            save_json=True,  # 保存JSON格式的结果
            plots=True,  # 生成评估图表
            project=results_path,  # 将结果保存到results_path
            name='test_evaluation',  # 子目录名称
            exist_ok=True,
            verbose=True
        )
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        raise
    
    # 对测试集进行预测并保存带检测框的图片
    print("\n开始对测试集进行预测并保存结果图片...")
    try:
        # 获取测试集所有图片路径
        test_image_files = []
        if os.path.exists(test_images_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                test_image_files.extend(Path(test_images_dir).glob(ext))
                test_image_files.extend(Path(test_images_dir).glob(ext.upper()))
        
        if test_image_files:
            print(f"找到 {len(test_image_files)} 张测试图片")
            # 使用predict方法对测试集进行预测
            results = model.predict(
                source=str(test_images_dir),  # 测试集图片目录
                imgsz=640,
                conf=0.25,  # 置信度阈值
                iou=0.7,    # IoU阈值
                device='0' if torch.cuda.is_available() else 'cpu',
                workers=0,  # 设置为0避免共享内存不足的问题
                save=True,  # 保存预测结果图片
                save_txt=True,  # 保存检测框坐标文本文件
                save_conf=True,  # 保存置信度
                project=results_path,  # 将结果保存到results_path
                name='test_predictions',  # 子目录名称
                exist_ok=True,
                verbose=True
            )
            print(f"预测结果已保存到: {os.path.join(results_path, 'test_predictions')}")
        else:
            print(f"警告: 在 {test_images_dir} 中未找到测试图片")
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        print("继续处理评估指标...")
    
    # 提取评估指标（兼容不同版本的YOLO）
    results_dict = {}
    
    # 辅助函数：安全地将值转换为float（处理数组和标量）
    def safe_float(value, default=0.0):
        """安全地将值转换为float，处理数组和标量"""
        if value is None:
            return default
        try:
            if isinstance(value, (list, np.ndarray)):
                # 如果是数组，取平均值
                return float(np.mean(value))
            else:
                return float(value)
        except (ValueError, TypeError):
            return default
    
    # 尝试提取基本指标
    if hasattr(metrics, 'box'):
        box_metrics = metrics.box
        
        # 提取总体指标（所有类别的平均值）
        # 如果值是数组，第一个元素通常是总体平均值
        if hasattr(box_metrics, 'p'):
            p_value = box_metrics.p
            if isinstance(p_value, (list, np.ndarray)):
                # 如果是数组，第一个元素是总体平均值
                results_dict['metrics/precision(B)'] = safe_float(p_value[0] if len(p_value) > 0 else p_value)
            else:
                results_dict['metrics/precision(B)'] = safe_float(p_value)
        
        if hasattr(box_metrics, 'r'):
            r_value = box_metrics.r
            if isinstance(r_value, (list, np.ndarray)):
                results_dict['metrics/recall(B)'] = safe_float(r_value[0] if len(r_value) > 0 else r_value)
            else:
                results_dict['metrics/recall(B)'] = safe_float(r_value)
        
        if hasattr(box_metrics, 'map50'):
            map50_value = box_metrics.map50
            if isinstance(map50_value, (list, np.ndarray)):
                results_dict['metrics/mAP50(B)'] = safe_float(map50_value[0] if len(map50_value) > 0 else map50_value)
            else:
                results_dict['metrics/mAP50(B)'] = safe_float(map50_value)
        
        if hasattr(box_metrics, 'map'):
            map_value = box_metrics.map
            if isinstance(map_value, (list, np.ndarray)):
                results_dict['metrics/mAP50-95(B)'] = safe_float(map_value[0] if len(map_value) > 0 else map_value)
            else:
                results_dict['metrics/mAP50-95(B)'] = safe_float(map_value)
        
        # 提取每个类别的指标
        class_names = list(original_config['names'].values())
        
        # 提取每个类别的mAP50-95
        if hasattr(box_metrics, 'maps') and box_metrics.maps is not None:
            maps = box_metrics.maps if isinstance(box_metrics.maps, np.ndarray) else np.array(box_metrics.maps)
            # maps数组通常第一个元素是总体平均值，后面是每个类别的值
            # 但根据YOLO输出，maps可能直接是每个类别的值
            for i, class_name in enumerate(class_names):
                if i < len(maps):
                    results_dict[f'metrics/mAP50-95(B)_{class_name}'] = safe_float(maps[i])
        
        # 提取每个类别的precision
        if hasattr(box_metrics, 'p') and isinstance(box_metrics.p, (list, np.ndarray)):
            p_values = np.array(box_metrics.p)
            # p_values数组第一个元素是总体平均值，后面是每个类别的值
            for i, class_name in enumerate(class_names):
                idx = i + 1 if len(p_values) > len(class_names) else i  # 如果第一个是总体值，从索引1开始
                if idx < len(p_values):
                    results_dict[f'metrics/precision(B)_{class_name}'] = safe_float(p_values[idx])
        
        # 提取每个类别的recall
        if hasattr(box_metrics, 'r') and isinstance(box_metrics.r, (list, np.ndarray)):
            r_values = np.array(box_metrics.r)
            for i, class_name in enumerate(class_names):
                idx = i + 1 if len(r_values) > len(class_names) else i
                if idx < len(r_values):
                    results_dict[f'metrics/recall(B)_{class_name}'] = safe_float(r_values[idx])
        
        # 提取每个类别的mAP50
        if hasattr(box_metrics, 'map50') and isinstance(box_metrics.map50, (list, np.ndarray)):
            map50_values = np.array(box_metrics.map50)
            for i, class_name in enumerate(class_names):
                idx = i + 1 if len(map50_values) > len(class_names) else i
                if idx < len(map50_values):
                    results_dict[f'metrics/mAP50(B)_{class_name}'] = safe_float(map50_values[idx])
    else:
        # 如果metrics结构不同，尝试直接访问
        print("警告: metrics结构可能与预期不同，尝试直接提取...")
        if hasattr(metrics, 'results_dict'):
            results_dict = metrics.results_dict
        else:
            # 保存原始metrics信息用于调试
            results_dict['warning'] = '无法提取标准指标，请检查metrics对象结构'
            print(f"metrics对象属性: {dir(metrics)}")
    
    # 保存评估结果到JSON文件
    metrics_json_path = os.path.join(results_path, 'test_metrics.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存到: {metrics_json_path}")
    
    # 保存评估结果到文本文件
    metrics_txt_path = os.path.join(results_path, 'test_metrics.txt')
    with open(metrics_txt_path, 'w', encoding='utf-8') as f:
        f.write("YOLO模型测试集评估结果\n")
        f.write("=" * 50 + "\n\n")
        for metric_name, value in results_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric_name}: {value:.4f}\n")
            else:
                f.write(f"{metric_name}: {value}\n")
    print(f"评估指标文本已保存到: {metrics_txt_path}")
    
    # 打印评估结果
    print("\n测试集评估结果:")
    print("=" * 50)
    for metric_name, value in results_dict.items():
        if isinstance(value, (int, float)):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    print("=" * 50)
    
    return results_dict


def train_target_detection(yaml_path: str, model_path: str, results_path: str, time_limit: int = 600):
    """
    训练或加载YOLO目标检测模型，并在测试集上评估
    
    Args:
        yaml_path: 数据集配置文件路径
        model_path: 模型保存路径
        results_path: 结果保存路径
        time_limit: 训练时间限制（未使用，保留以兼容接口）
    """
    os.makedirs(results_path, exist_ok=True)
    
    # 确定模型权重路径
    model_weights_path = os.path.join(model_path, 'breast_cancer_detection', 'weights', 'best.pt')
    
    # 测试集路径
    dataset_base_dir = os.path.dirname(yaml_path)
    test_images_dir = os.path.join(dataset_base_dir, 'images', 'test')
    test_labels_dir = os.path.join(dataset_base_dir, 'labels', 'test')
    
    # 检查模型是否存在
    if os.path.exists(model_weights_path):
        print(f"发现已存在的模型，加载模型: {model_weights_path}")
        model = YOLO(model_weights_path)
        print("模型加载成功！")
    else:
        print(f"模型不存在，开始训练新模型...")
        print(f"模型将保存到: {model_weights_path}")
        
        # 加载预训练模型
        model = YOLO('models/yolo11x.pt')  # 使用YOLOv11-x版本
        
        # 训练模型
        results = model.train(
            data=yaml_path,  # 数据集配置文件
            epochs=300,                        # 训练轮数
            imgsz=640,                         # 图像大小
            batch=8,                          # 批次大小
            device='0' if torch.cuda.is_available() else 'cpu',  # 使用GPU如果可用
            workers=0,                         # 数据加载的工作进程数
            patience=50,                       # 早停的耐心值
            save=True,                         # 保存模型
            project=model_path,              # 项目目录
            name='breast_cancer_detection',    # 实验名称
            exist_ok=True,                     # 允许覆盖已存在的实验目录
            pretrained=True,                   # 使用预训练权重
            optimizer='auto',                  # 自动选择优化器
            verbose=True,                      # 显示详细信息
            seed=42,                           # 随机种子
            deterministic=True,                # 确定性训练
        )
        
        # 在验证集上验证模型
        print("在验证集上验证模型...")
        model.val()
        
        # 重新加载最佳模型
        model = YOLO(model_weights_path)
        print("训练完成，已加载最佳模型！")
    
    # 在测试集上评估模型
    print("\n" + "=" * 50)
    print("开始在测试集上评估模型...")
    print("=" * 50)
    
    # 检查测试集目录是否存在
    if not os.path.exists(test_images_dir):
        print(f"警告: 测试集图像目录不存在: {test_images_dir}")
        print("跳过测试集评估")
        return model
    
    if not os.path.exists(test_labels_dir):
        print(f"警告: 测试集标签目录不存在: {test_labels_dir}")
        print("跳过测试集评估")
        return model
    
    # 执行评估
    test_metrics = evaluate_yolo_model(model, yaml_path, test_images_dir, test_labels_dir, results_path)
    
    print(f"\n所有结果已保存到: {results_path}")
    
    return model

if __name__ == "__main__":
    COMMON_TRAIN_ARGS = dict(
        num_workers_train=DEFAULT_TRAIN_NUM_WORKERS,
        num_workers_inference=DEFAULT_INFER_NUM_WORKERS,
        per_gpu_batch_size=DEFAULT_PER_GPU_BATCH_SIZE,
    )
    # # # eye disease
    # train_model_DR(train_data_path="data/Img/eye_disease/train_DR.csv",test_data_path="data/Img/eye_disease/test_DR.csv", model_path="models/eye_disease/DR",results_path="results/eye_disease/DR", time_limit=600, **COMMON_TRAIN_ARGS)
    # train_model_DR(train_data_path="data/Img/eye_disease/train_MH.csv",test_data_path="data/Img/eye_disease/test_MH.csv", model_path="models/eye_disease/MH",results_path="results/eye_disease/MH", time_limit=600, **COMMON_TRAIN_ARGS)
    # train_model_DR(train_data_path="data/Img/eye_disease/train_DN.csv",test_data_path="data/Img/eye_disease/test_DN.csv", model_path="models/eye_disease/DN",results_path="results/eye_disease/DN", time_limit=600, **COMMON_TRAIN_ARGS)
    # train_model_DR(train_data_path="data/Img/eye_disease/train_ODC.csv",test_data_path="data/Img/eye_disease/test_ODC.csv", model_path="models/eye_disease/ODC",results_path="results/eye_disease/ODC", time_limit=600, **COMMON_TRAIN_ARGS)
    # train_model_DR(train_data_path="data/Img/eye_disease/train_TSLN.csv",test_data_path="data/Img/eye_disease/test_TSLN.csv", model_path="models/eye_disease/TSLN",results_path="results/eye_disease/TSLN", time_limit=600, **COMMON_TRAIN_ARGS)
    # train_model_DR(train_data_path="data/Img/eye_disease/train_Normal.csv",test_data_path="data/Img/eye_disease/test_Normal.csv", model_path="models/eye_disease/Normal",results_path="results/eye_disease/Normal", time_limit=600, **COMMON_TRAIN_ARGS)
    # # # skin cancer
    # train_model_DR(train_data_path="data/Img/Skin_Cancer/train_dataset.csv",test_data_path="data/Img/Skin_Cancer/test_dataset.csv", model_path="models/Skin_Cancer",results_path="results/Skin_Cancer", time_limit=600, **COMMON_TRAIN_ARGS)
    # # # breast cancer
    # 目标检测
    train_target_detection(yaml_path="data/Img/Breast_cancer/yolo_dataset/dataset.yaml", model_path="models/breast_cancer_detection",results_path="results/breast_cancer_detection", time_limit=600)
    # chest cancer
    # train_model_DR(train_data_path="data/Img/chest_cancer/train_data.csv",test_data_path="data/Img/chest_cancer/test_data.csv", model_path="models/chest_cancer_detection",results_path="results/chest_cancer_detection", time_limit=600, **COMMON_TRAIN_ARGS)