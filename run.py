import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
from dotenv import load_dotenv
from app.app import create_app
import shap
shap.initjs()


load_dotenv()
# config = os.getenv('FLASK_ENV') or 'development'

config = 'development'
# config = 'production'

# Ensure the environment is passed to subprocess
env = os.environ.copy()

# vllm_cmd = (
#     "CUDA_VISIBLE_DEVICES=0 vllm serve /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/MedicalExaminationAgent/PhysicalExaminationAgent/client/model/Jianxiaozhi-14B-instruct "
#     "--trust-remote-code --load-format=bitsandbytes --quantization=bitsandbytes"
# )
# vllm_process = subprocess.Popen(vllm_cmd, shell=True)

app = create_app(config)
def print_routes(app):
    print("=== ROUTES ===")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        methods = ",".join(sorted(m for m in rule.methods if m not in {"HEAD", "OPTIONS"}))
        print(f"{rule.rule:50s}  ->  {rule.endpoint:30s} [{methods}]")

print_routes(app)

if __name__ == "__main__":
    try:
        if config == 'development':
            app.run(debug=True,use_reloader=False)
        else:
            from werkzeug.serving import run_simple
            run_simple('0.0.0.0', 52315, app)
    finally:
        print("vllm_process.terminate()")
        # vllm_process.terminate()
        # vllm_process.wait()
