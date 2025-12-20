import ray
ray.shutdown()
ray.init(num_cpus=192, num_gpus=1) 
import json
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set matplotlib font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical symbols
quickVerification=False

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    title="Confusion Matrix",
    save_path="confusion_matrix.png",
    normalize=False,
    cmap="Blues",
    annot=True
):
    """
    绘制美化后的混淆矩阵图
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='gray',
        square=True,
        cbar=True
    )

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存混淆矩阵数据为JSON
    json_path = save_path.replace('.png', '_data.json')
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'normalize': normalize,
        'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true),
        'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cm_data, f, ensure_ascii=False, indent=4)
    
    return cm

def plot_roc_curves(y_true, y_prob, labels, title="ROC Curves", save_path="roc_curves.png"):
    """
    绘制美观的多类或二类 ROC 曲线图，二分类只画正类且AUC与控制台一致。
    """
    plt.figure(figsize=(6, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    colors = plt.get_cmap("tab10")
    
    # 保存ROC曲线数据
    roc_data = {
        'labels': labels,
        'curves': []
    }

    # 二分类只画正类，且保证AUC与控制台一致
    # if len(labels) == 2:
    #     y_true_bin = pd.Series(y_true).astype(int)
    #     y_score = y_prob.iloc[:, 1]
    #     fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    #     roc_auc = auc(fpr, tpr)
    #     plt.plot(
    #         fpr,
    #         tpr,
    #         label=f"{labels[1]} (AUC = {roc_auc:.2f})",
    #         linewidth=2,
    #         color=colors(1)
    #     )
    # else:
    for i, label in enumerate(labels):
        fpr, tpr, thresholds = roc_curve((y_true == i).astype(int), y_prob.iloc[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{label} (AUC = {roc_auc:.2f})",
            linewidth=2,
            color=colors(i)
        )
        # 保存每条ROC曲线的数据
        roc_data['curves'].append({
            'label': label,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        })

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(title, fontsize=16, pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 保存ROC曲线数据为JSON
    json_path = save_path.replace('.png', '_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(roc_data, f, ensure_ascii=False, indent=4)

def plot_regression_results(y_true, y_pred, title_prefix, save_dir):
    """
    绘制回归任务的真实值-预测值散点图和残差图。
    """
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='royalblue', edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values', fontsize=13)
    plt.ylabel('Predicted Values', fontsize=13)
    plt.title(f'{title_prefix} True vs Predicted', fontsize=16, pad=12)
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, f'{title_prefix}_true_vs_pred.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 残差图
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, color='darkorange', edgecolor='k')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Values', fontsize=13)
    plt.ylabel('Residuals (True - Pred)', fontsize=13)
    plt.title(f'{title_prefix} Residual Plot', fontsize=16, pad=12)
    plt.tight_layout()
    residual_path = os.path.join(save_dir, f'{title_prefix}_residuals.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存回归数据为JSON
    regression_data = {
        'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true),
        'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
        'residuals': residuals.tolist() if hasattr(residuals, 'tolist') else list(residuals)
    }
    json_path = os.path.join(save_dir, f'{title_prefix}_regression_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(regression_data, f, ensure_ascii=False, indent=4)

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories if they don't exist
    os.makedirs(config['model_path'], exist_ok=True)
    os.makedirs(config['result_path'], exist_ok=True)
    
    # Load data
    df = pd.read_csv(config['data_path'])
    
    # Prepare features and target
    feature_columns = list(config['feature_columns'].keys())
    feature_columns.remove(config['target_column'])
    
    # Split data into train and test sets
    if config['model_type'] == 'binary':
        # Use stratified sampling for binary classification
        train_data, test_data = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[config['target_column']]  # Ensure same class ratio in train and test
        )
    else:
        train_data = df.sample(frac=0.8, random_state=42)
        test_data = df.drop(train_data.index)
    
    # Initialize and train AutoGluon model
    if config['model_type'] == 'regression':
        eval_metric = 'r2'
    elif config['model_type'] == 'binary':
        eval_metric = 'roc_auc'
    else:
        eval_metric = 'accuracy'
    predictor = TabularPredictor(
        label=config['target_column'],
        path=config['model_path'],
        problem_type=config['model_type'],
        eval_metric=eval_metric  # Primary metric for model selection
    )
    
    # Train the model with best model selection
    if quickVerification:
        predictor.fit(
            train_data=train_data,
            time_limit=60,  # 1 minute training time
            presets='optimize_for_deployment'
        )
    else:
        predictor.fit(
            train_data=train_data,
            time_limit=3600,  # 1 hour training time
            presets='best_quality',  # Use best quality preset
            num_bag_folds=5,  # Use 5-fold cross-validation
            num_stack_levels=2,  # Use 2 levels of stacking
            verbosity=2  # Show detailed training progress
        )
    
    # Get predictions and probabilities
    y_pred = predictor.predict(test_data)
    if config['model_type'] in ['binary', 'multiclass']:
        y_prob = predictor.predict_proba(test_data)
    else:
        y_prob = None
    
    # 保存预测结果数据为JSON（用于调试绘图）
    predictions_data = {
        'y_true': test_data[config['target_column']].tolist() if hasattr(test_data[config['target_column']], 'tolist') else list(test_data[config['target_column']]),
        'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
    }
    if y_prob is not None:
        # 处理pandas DataFrame或numpy array
        if hasattr(y_prob, 'values'):
            predictions_data['y_prob'] = y_prob.values.tolist()
        elif hasattr(y_prob, 'tolist'):
            predictions_data['y_prob'] = y_prob.tolist()
        else:
            predictions_data['y_prob'] = list(y_prob)
        if config['model_type'] in ['binary', 'multiclass']:
            predictions_data['class_labels'] = predictor.class_labels.tolist() if hasattr(predictor.class_labels, 'tolist') else list(predictor.class_labels)
    
    predictions_json_path = os.path.join(config['result_path'], f"{config['model_name']}_predictions.json")
    with open(predictions_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=4)
    
    # Calculate various metrics
    if config['model_type'] == 'regression':
        y_true = test_data[config['target_column']]
        y_pred_reg = predictor.predict(test_data)
        n = len(y_true)
        p = len(feature_columns)
        mse = mean_squared_error(y_true, y_pred_reg)
        mae = mean_absolute_error(y_true, y_pred_reg)
        r2 = r2_score(y_true, y_pred_reg)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float('nan')
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Adj-R2': adj_r2
        }
    else:
        metrics = {
            'Accuracy': accuracy_score(test_data[config['target_column']], y_pred),
            'Balanced Accuracy': balanced_accuracy_score(test_data[config['target_column']], y_pred),
            'F1 Macro': f1_score(test_data[config['target_column']], y_pred, average='macro'),
            'Precision Macro': precision_score(test_data[config['target_column']], y_pred, average='macro'),
            'Recall Macro': recall_score(test_data[config['target_column']], y_pred, average='macro'),
        }
    
    # Calculate AUC for each class
    if config['model_type'] == 'multiclass':
        auc_scores = {}
        average_auc = 0
        for i, class_name in enumerate(predictor.class_labels):
            auc_scores[f'AUC_{class_name}'] = roc_auc_score(
                test_data[config['target_column']] == i,
                y_prob.iloc[:, i]
            )
            average_auc += auc_scores[f'AUC_{class_name}']
        average_auc /= len(predictor.class_labels)
        metrics['Average AUC'] = average_auc
    elif config['model_type'] == 'binary':
        metrics['AUC'] = roc_auc_score(test_data[config['target_column']], y_prob.iloc[:, 1])
    
    # Get feature importance
    feature_importance = predictor.feature_importance(test_data)
    
    # Prepare results dictionary
    results = {
        'metrics': metrics,
        'feature_importance': feature_importance.to_dict(),
        'model_info': {
            'model_path': config['model_path'],
            'model_name': config['model_name'],
            'model_type': config['model_type'],
            'target_column': config['target_column'],
            'feature_columns': feature_columns
        }
    }
    
    # Save evaluation results as JSON
    save_path = os.path.join(config['result_path'], f"{config['model_name']}_results.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Create visualizations
    if config['model_type'] == 'multiclass':
        # Get class labels from target_mapping_draw
        class_labels = [config['target_mapping_draw'][str(i)] for i in range(len(config['target_mapping_draw']))]
        
        # Plot confusion matrix
        cm_path = os.path.join(config['result_path'], f"{config['model_name']}_confusion_matrix.png")
        plot_confusion_matrix(
            test_data[config['target_column']],
            y_pred,
            class_labels,
            f'Confusion Matrix - {config["model_name"]}',
            cm_path
        )
        
        # Plot ROC curves
        roc_path = os.path.join(config['result_path'], f"{config['model_name']}_roc_curves.png")
        plot_roc_curves(
            test_data[config['target_column']],
            y_prob,
            class_labels,
            f'ROC Curves - {config["model_name"]}',
            roc_path
        )
    elif config['model_type'] == 'binary':
        # Get class labels from target_mapping_draw
        class_labels = [config['target_mapping_draw']['0'], config['target_mapping_draw']['1']]
        
        # Plot confusion matrix
        cm_path = os.path.join(config['result_path'], f"{config['model_name']}_confusion_matrix.png")
        plot_confusion_matrix(
            test_data[config['target_column']],
            y_pred,
            class_labels,
            f'Confusion Matrix - {config["model_name"]}',
            cm_path
        )
        
        # Plot ROC curve
        roc_path = os.path.join(config['result_path'], f"{config['model_name']}_roc_curve.png")
        plot_roc_curves(
            test_data[config['target_column']],
            y_prob,
            class_labels,
            f'ROC Curve - {config["model_name"]}',
            roc_path
        )
    elif config['model_type'] == 'regression':
        # 回归可视化
        plot_regression_results(
            y_true,
            y_pred_reg,
            title_prefix=config['model_name'],
            save_dir=config['result_path']
        )
    
    # Print results
    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print(f"\nModel saved to: {config['model_path']}")
    print(f"Results saved to: {save_path}")
    if config['model_type'] == 'multiclass':
        print(f"Confusion matrix saved to: {cm_path}")
        print(f"ROC curves saved to: {roc_path}")

if __name__ == "__main__":
    # 糖尿病
    main(config_path='risk_assessment/data/config/diabetes.json')
    # 肥胖
    main(config_path='risk_assessment/data/config/obesity.json')
    # 产妇风险
    main(config_path='risk_assessment/data/config/maternal_health.json')
    # 肝炎
    main(config_path='risk_assessment/data/config/hepatitis.json')
    # 心脏病
    main(config_path='risk_assessment/data/config/heart_risk.json')
    # 心力衰竭
    main(config_path='risk_assessment/data/config/heart_failure_clinical_records.json')
    # 早期糖尿病
    main(config_path='risk_assessment/data/config/early_diabetes.json')
    # 心脏病
    main(config_path='risk_assessment/data/config/heart_disease.json')
    # 中风风险
    main(config_path='risk_assessment/data/config/stroke_risk.json')
    # 脱发
    main(config_path='risk_assessment/data/config/hair_fall.json')
    # 高血压
    main(config_path='risk_assessment/data/config/hypertension_risk.json')
    # 睡眠障碍
    main(config_path='risk_assessment/data/config/sleep_disorder.json')
    # 肺癌
    main(config_path='risk_assessment/data/config/lung_cancer.json')
    # 物理检查糖尿病
    main(config_path='risk_assessment/data/config/physical_examination_diabetes.json')
    # 心率
    main(config_path='risk_assessment/data/config/calories.json')
    # 乳腺癌
    main(config_path='risk_assessment/data/config/breast_cancer.json')
    # 肾病
    main(config_path='risk_assessment/data/config/kidney_disease.json')
    # 甲状腺
    main(config_path='risk_assessment/data/config/thyroid_diff.json')
    # 前列腺癌
    main(config_path='risk_assessment/data/config/prostate_cancer.json')
    # 鼻咽癌
    main(config_path='risk_assessment/data/config/nasopharyngeal_cancer.json')
    # 宫颈癌
    main(config_path='risk_assessment/data/config/cervical_cancer.json')
    # 直肠癌
    main(config_path='risk_assessment/data/config/colorectal_cancer.json')
    # 胰腺癌
    main(config_path='risk_assessment/data/config/pancreatic_cancer.json')