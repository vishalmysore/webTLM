"""Create the HF repo (if needed) and upload this folder's contents —
pure Python API, so it works even when huggingface-cli isn't on PATH.

Usage:
    python push_to_hf.py                 # uploads to VishalMysore/webLTM
    python push_to_hf.py your-username    # or a different namespace

Requires you to already be logged in once (huggingface-cli login, or
`python -m huggingface_hub.commands.huggingface_cli login`, or set the
HF_TOKEN environment variable) — same login cookgptlama used.
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

username = sys.argv[1] if len(sys.argv) > 1 else "VishalMysore"
repo_id = f"{username}/webLTM"
folder = Path(__file__).parent

api = HfApi()
create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=str(folder),
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=["*.md", "*.json", "*.py", "*.safetensors"],
)
print(f"done: https://huggingface.co/{repo_id}")
