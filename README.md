# JiShi：Multi-agent Medical Language Models for Health Examination Risk Assessment

## Overview

This project implements a comprehensive multi-agent medical system for health examination risk assessment. The system leverages large language models (Qwen3 32B and Qwen3 14B) trained on a curated corpus of 2.81 million medical dialogues to provide intelligent health examination report interpretation and disease risk assessment across 20 disease categories.

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
- **Qwen3 Models**: 32B and 14B parameter models fine-tuned for medical tasks
- **Three-Stage Training Strategy**:
  1. Supervised fine-tuning on 2.81M medical dialogues
  2. Direct preference optimization with medical questions
  3. Task-specific fine-tuning for health management
- **PPMSA Factory**: Framework for generating report-centered dialogues

### 🔬 Risk Stratification
- Quantitative disease risk assessment across 20 categories
- Evidence-based recommendations
- Personalized health management suggestions
- Closed-loop system from data perception to individualized management

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

## Usage

### Starting the Main Application

1. **Start the risk assessment service** (in a separate terminal):
   ```bash
   cd risk_assessment
   python app.py
   ```
   This starts the risk assessment service on port 5002.

2. **Start the main Flask application**:
   ```bash
   python run.py
   ```
   The application will be available at `http://localhost:52315` (development) or `http://0.0.0.0:52315` (production).

### Starting MCP Health Server (Optional)

For advanced medical analysis capabilities:

```bash
cd MCP/MCP-Health
python server.py
```

Or using SSE version:
```bash
python server_sse.py --transport sse --port 8088
```

### API Endpoints

The main application provides several API endpoints:

- **Health Examination Report Upload**: Upload and analyze health examination reports
- **Risk Assessment**: Get disease risk assessments based on user dialogues and reports
- **Chat Interface**: Multi-turn conversations with medical AI agents
- **File Management**: Upload, store, and retrieve medical reports

See the Swagger documentation at `/api/docs` for detailed API reference.

## Project Structure

```
client/
├── app/                    # Main Flask application
│   ├── app.py             # Flask app factory
│   ├── config/            # Configuration files
│   ├── db/                # Database models and setup
│   ├── modules/           # Application modules
│   │   ├── auth/         # Authentication
│   │   ├── chat/         # Chat interface
│   │   └── files/        # File management
│   └── util/              # Utility functions
│       ├── agent_config.py    # Multi-agent configuration
│       ├── clinical_analyst.py # Clinical language analysis
│       └── detectron2/    # Object detection utilities
├── risk_assessment/       # Risk assessment service
│   ├── app.py            # Risk assessment Flask service
│   ├── train_model.py    # Model training scripts
│   └── requirements.txt   # Risk assessment dependencies
├── MCP/                   # Model Context Protocol servers
│   ├── MCP-Health/       # Healthcare MCP server
│   ├── ollama-mcp/       # Ollama MCP client
│   └── healthcare-mcp-public/ # Public healthcare MCP
├── model/                 # Trained model files
├── run.py                 # Main application entry point
├── environment.yml        # Conda environment specification
└── README.md             # This file
```

## Training and Model Development

## Disease Categories Supported

The system provides risk assessment for 20 disease categories:

1. Calorie Intake Assessment
2. Diabetes
3. Obesity
4. Early Diabetes Detection
5. Hair Loss
6. Heart Disease
7. Heart Failure
8. Heart Risk
9. Hepatitis
10. Hypertension Risk
11. Lung Cancer
12. Maternal Health
13. Physical Examination Diabetes
14. Sleep Disorders
15. Stroke Risk
16. Chronic Kidney Disease
17. Thyroid Cancer
18. Brain Stroke
19. And additional categories...

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

