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
            # Use Waitress as production WSGI server (no warning)
            try:
                from waitress import serve
                print(f"\n{'='*60}")
                print(f"Starting Waitress server on http://0.0.0.0:52315")
                print(f"Press CTRL+C to stop the server")
                print(f"{'='*60}\n")
                serve(app, host='0.0.0.0', port=52315, _quiet=False)
            except ImportError:
                print("ERROR: waitress is not installed. Please run: pip install waitress")
                print("Falling back to Werkzeug development server...")
                from werkzeug.serving import run_simple
                run_simple('0.0.0.0', 52315, app)
            except KeyboardInterrupt:
                print("\n\nServer stopped by user")
    finally:
        print("vllm_process.terminate()")
        # vllm_process.terminate()
        # vllm_process.wait()
