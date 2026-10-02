"""legal_ft: QLoRA fine-tuning + before/after evaluation on CUAD."""

from legal_ft.config import load_dotenv

# Before any submodule imports transformers/huggingface_hub, which read HF_HOME once at import.
load_dotenv()

__version__ = "0.1.0"
