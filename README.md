# TruthLens AI - Advanced Fake News Detection System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-brightgreen?style=for-the-badge&logo=render)](https://fake-news-detection-w7kw.onrender.com)

🌐 **Live Demo Application**: [https://fake-news-detection-w7kw.onrender.com](https://fake-news-detection-w7kw.onrender.com)

TruthLens AI is a high-fidelity, production-grade fake news detection platform. It leverages a state-of-the-art hybrid ensemble architecture combining Transformer models with Recurrent Neural Networks and classical linguistic analysis to provide explainable and highly accurate credibility assessments.

## 🚀 Key Features

- **Hybrid Ensemble Architecture**: Combines BERT (contextual) and BiLSTM (sequential) models for robust prediction.
- **Explainable AI (XAI)**: Integrated attention mechanism visualizes token-level importance, highlighting exactly which words triggered a "Fake" classification.
- **Neural Pipeline Visualization**: Watch the real-time processing flow from preprocessing to final calibration in a cinematic dashboard.
- **Production-Grade Backend**: FastAPI-based server with Pydantic configuration, robust logging, and graceful lifespan management.
- **Dockerized Deployment**: Fully containerized environment with Gunicorn/Uvicorn workers for high availability.
- **Linguistic Heuristics**: Custom calibration using journalistic trust markers and sensationalism scoring to reduce false positives.
- **Premium Glassmorphic UI**: A high-end, responsive dashboard featuring dark mode, neural backgrounds, and fluid animations.

## 🏗️ Model Architecture

### 1. BiLSTM with Self-Attention (DL Objective — CSR311)
A Bidirectional LSTM network (hidden=256, 2 layers, dropout=0.3) with a **Scaled Dot-Product Self-Attention** layer that provides transparency by assigning weights to each token.

- **Loss**: BCELoss  
- **Optimizer**: Adam  
- **Tokenization**: NLTK `word_tokenize`, padded to 200 tokens  
- **Comparison**: BiLSTM+Attention vs BiLSTM (No Attention) — reported via Accuracy and AUC-ROC

### 2. Hybrid BERT Model (NLP Objective — CSR322)
A BERT-based transformer model (`bert-base-uncased`) augmented with a classical feature layer:
- **TF-IDF Vectorization**: n-gram (1,2) statistical word importance
- **POS Analysis**: Adjective-to-noun ratio via NLTK POS tagging
- **Sentiment Analysis**: VADER compound, positive, negative, neutral scores

**Three-way comparison** reported via F1:
- (a) BERT alone
- (b) Classical features alone
- (c) Hybrid (BERT [CLS] + Classical features)

### 3. Smart Calibration Ensemble
The system uses a weighted ensemble (60% BERT / 40% BiLSTM) dynamically adjusted by heuristic scores:
- **Trust Markers**: Keywords typical of credible journalism
- **Sensationalism Markers**: Phrases often used in clickbait and misinformation

## 📊 Dataset

The models are trained on the **WELFake Dataset**, a comprehensive collection of **72,000+ news articles** categorized as "Real" (label=0) or "Fake" (label=1). The dataset combines articles from four sources: Kaggle, McIntire, Reuters, and BuzzFeed Political.

| Property | Value |
|----------|-------|
| Total Articles | 72,134 |
| Fake Articles | ~35,000 |
| Real Articles | ~37,000 |
| Features | Title, Text, Label |
| Format | CSV |

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Deep Learning | PyTorch | Model definition, training, inference |
| Transformers | HuggingFace | Pre-trained BERT model and tokenizer |
| NLP | NLTK | Tokenization, POS tagging, VADER sentiment |
| ML Utilities | Scikit-Learn | TF-IDF vectorization, train/test split |
| Visualization | Matplotlib, Seaborn | Attention heatmap plots |
| Backend API | FastAPI + Pydantic | Async REST API, input validation |
| Frontend | HTML5, CSS3, Vanilla JS | Glassmorphism UI, attention heatmap rendering |
| DevOps | Docker + Docker Compose | Containerization |

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Fake News Detection"
```

### 2. Standard Local Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger vader_lexicon stopwords wordnet
```

### 3. Run the Full Experiment
This trains all models, runs comparisons, generates heatmaps, and produces error analysis:
```bash
python main.py
```

**Output files** (saved to `output/`):
- `bilstm_attn.pth` — Trained BiLSTM + Attention model
- `bert_hybrid.pth` — Trained Hybrid BERT model
- `word_to_idx.pkl` — Vocabulary mapping
- `tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer
- `error_analysis.csv` — 20 misclassified examples with failure modes
- `experiment_results.txt` — Full results summary
- `attention_heatmap_fake.png` — Heatmap for a correctly classified Fake article
- `attention_heatmap_real.png` — Heatmap for a correctly classified Real article
- `attention_heatmap_misclassified.png` — Heatmap for a misclassified article

### 4. Run the BiLSTM Training Only (DL Objective)
```bash
python src/train.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Then open `http://localhost:8000` in your browser.

### 6. Running with Docker (Production)
```bash
docker-compose up --build
```

## 📈 Monitoring & Logs
The system generates an `app.log` file in the root directory:
```bash
tail -f app.log
```
Health endpoint: `http://localhost:8000/health`

## 📁 Project Structure

```
truthlens-ai/
├── app.py                      # FastAPI entry point, routing, lifespan model loading
├── config.py                   # Pydantic settings and configuration
├── main.py                     # Full experiment runner (both objectives)
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile                  # Container build specification
├── requirements.txt            # Python dependencies
├── WELFake_Dataset.csv         # Training dataset
├── src/
│   ├── bilstm_attention.py     # BiLSTM + Scaled Dot-Product Attention (PyTorch)
│   ├── bert_hybrid.py          # Hybrid BERT + classical features model (PyTorch)
│   ├── feature_extractor.py    # Classical NLP features (TF-IDF, POS, Sentiment)
│   ├── data_loader.py          # Dataset ingestion, NLTK tokenization, padding
│   ├── train.py                # Standalone DL Objective training script
│   ├── predict.py              # Interactive PyTorch prediction CLI
│   ├── utils.py                # Text preprocessing utilities
│   └── visualize_attention.py  # Attention heatmap generation (Matplotlib)
├── output/                     # Saved models, results, and heatmaps
└── frontend/
    └── index.html              # Glassmorphism UI with attention visualization
```

---
*Developed with excellence in AI and Design.*