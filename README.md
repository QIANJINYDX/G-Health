<h1>
  <img src="app/static/images/logo.png" alt="JiShi logo" width="40" style="vertical-align:middle; margin-right:10px;" />
  JiShi: Multi-agent Medical Language Models for Health Examination Risk Assessment
</h1>
## Overview

This project delivers a comprehensive multi-agent medical system for health examination risk assessment. It automatically interprets health check-up reports and generates structured, patient-friendly insights, including key abnormal findings, likely clinical implications, follow-up recommendations, and personalized risk stratification across 20 disease categories.

## Key Features

### 🏥 Multi-Agent Architecture
- **Clinical Language Analyst**: Analyzes user dialogues to determine relevant risk assessment models
- **Specialized Risk Assessment Agents**: 20 disease-specific risk assessment models including:
  - Diabetes (Type 1 & Early Detection)
  - Heart Disease & Heart Failure
  - Hypertension Risk
  - Obesity
  - Chronic Kidney Disease
  - Lung Cancer
  - Thyroid Cancer
  - Stroke Risk
  - Hepatitis
  - Sleep Disorders
  - Hair Loss
  - Maternal Health
  - And more...

### 📊 Health Examination Report Analysis
- Intelligent report structure parsing
- Abnormal indicator detection and interpretation
- Multi-turn conversation support for report-centered dialogues
- Personalized lifestyle and follow-up recommendations

### 🤖 Advanced AI Integration
We integrate Ollama for local LLM deployment and model invocation, providing reliable inference support for multi-agent report interpretation and risk assessment workflows.

### 🔬 Risk Stratification
- Quantitative disease risk assessment across 20 categories
- Evidence-based recommendations
- Personalized health management suggestions
- Closed-loop system from data perception to individualized management
- SHAP-based interpretability analysis

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for model inference)
- At least 50GB free disk space for models and dependencies
- Conda or Miniconda

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/QIANJINYDX/JiShi.git
   cd JiShi
   ```

2. **Install the JiShi conda environment**:
   ```bash
   conda create -n jishi python=3.11
   conda activate jishi
   ``` 
3 下载模型

4 下载表格数据
csv数据

5 配置mineru
modelscope download --model ppaanngggg/layoutreader --local_dir magic-pdf-models/layoutreader
modelscope download --model OpenDataLab/PDF-Extract-Kit-1.0 --local_dir magic-pdf-models/PDF-Extract-Kit-1.0
配置magic-pdf-models/magic-pdf.json目录下models-dir为PDF-Extract-Kit-1.0，layoutreader-model-dir为layoutreader
配置app/util/file_detection.py中的MINERU_TOOLS_CONFIG_JSON为绝对路径

## Usage

### Starting the Main Application

1. Start Ollama
   `ollama serve`

2. Start JiShi
   `python run.py`

3. Start the Risk Assessment module
   `python risk_assessment/app.py`

4. Start the RAG service (index/build module)
   `python app/util/rag_service.py --mode serve`

5. Start the MCP service modules
   `cd MCP`
   `python start_all_mcp.py`


## Training and Model Development

## Contributing

This is a research project. For contributions or questions, please contact the project maintainers.

## License

[Specify license here]

## Citation

If you use this work in your research, please cite:

```bibtex
@article{multiagent_medical_llm_2024,
  title={Multi-agent Medical Language Models for Health Examination Risk Assessment},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Acknowledgments

- Qwen team for the base language models
- All clinicians and medical assessors who contributed to evaluation
- The medical corpus contributors

## Contact

For questions or support, please contact the project team.

---

**Note**: This system is designed for research and clinical decision support. It should not replace professional medical judgment. Always consult qualified healthcare providers for medical decisions.

