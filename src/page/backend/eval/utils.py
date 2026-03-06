import threading
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

_model_cache = {}
_model_lock = threading.Lock()

def load_model(model_name="BAAI/bge-base-en-v1.5"):
    """
    Loads the model and tokenizer once and returns them (thread-safe).
    """
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:  # double-checked locking
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(device)
                model.eval()
                _model_cache[model_name] = (model, tokenizer, device)
    return _model_cache[model_name]

def generate_embedding(text, model, tokenizer, device):
    """
    Generates an embedding for the given text.
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0)
