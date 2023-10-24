from huggingface_hub import snapshot_download
from pathlib import Path
cache_path = Path.home() / ".cache/huggingface/transformers/models" / "opt-6.7b"
snapshot_download(repo_id="facebook/opt-6.7b", ignore_regex=["*.h5", "*.ot", "*.msgpack"], cache_dir=cache_path)