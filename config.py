import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "TruthLens AI"
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # Model Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    
    # Model Files
    WORD_TO_IDX_PATH: str = os.path.join(OUTPUT_DIR, "word_to_idx.pkl")
    BILSTM_MODEL_PATH: str = os.path.join(OUTPUT_DIR, "bilstm_attn.pth")
    BERT_MODEL_PATH: str = os.path.join(OUTPUT_DIR, "bert_hybrid.pth")
    TFIDF_MODEL_PATH: str = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
    
    # Security
    CORS_ORIGINS: List[str] = ["*"] # Change to specific domains in real production
    
    # Google Fact Check Tools API (free, get key from https://console.cloud.google.com/)
    # Leave empty to disable real-time fact-checking (app still works without it)
    GOOGLE_FACTCHECK_API_KEY: str = ""
    
    # Heuristics
    TRUST_MARKERS: List[str] = ["according to", "reported by", "stated that", "spokesperson", "official sources", "confirmed by", "citing"]
    SENSATIONAL_MARKERS: List[str] = ["shocker", "you won't believe", "exposed!", "conspiracy", "hidden truth", "they don't want you to know"]

    class Config:
        env_file = ".env"

settings = Settings()
