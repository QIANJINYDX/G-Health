<h1>
  <img src="app/static/images/logo.png" alt="JiShi logo" width="40" style="vertical-align:middle; margin-right:10px;" />
  G-Health: Clinically grounded multi-agent artificial intelligence for preventive health management
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
   git clone https://github.com/QIANJINYDX/G-Health.git
   cd JiShi
   ```

2. **Install the JiShi conda environment**:

   ```bash
   conda create -n ghealth python=3.11
   conda activate ghealth
   ```

3. **Download the risk assessment model**:

   ```bash
   ```

4. **Download the risk assessment configuration files**:

   ```bash
   ```

5. **Configure MinerU**:

   ```bash
   ```

6. **Install Ollama**:

   ```bash
   ```


## Usage

### Starting the Main Application

1. Start Ollama

```bash
ollama serve
```

2. Start G-health

```bash
python run.py
```

3. Start the Risk Assessment module

```bash
python risk_assessment/app.py
```

4. Start the RAG service (build/index module)

```bash
python app/util/rag_service.py --mode serve
```

5. Start the MCP service modules

```bash
cd MCP
python start_all_mcp.py
```


## Acknowledgments

- Qwen team for the base language models
- All clinicians and medical assessors who contributed to evaluation
- The medical corpus contributors


**Note**: This system is designed for research and clinical decision support. It should not replace professional medical judgment. Always consult qualified healthcare providers for medical decisions.

