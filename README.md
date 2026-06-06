# ScholarSync 🔍

> AI-powered research discovery platform - semantic search, topic classification, and paper recommendations over 476 arXiv papers.

---

## What is ScholarSync?

Most research search tools match keywords. ScholarSync matches **meaning**.

Search *"attention mechanism in transformers"* and it finds papers about BERT, self-attention, and language models - even if those exact words never appear. It understands what you're looking for, not just what you typed.

Built as an end-to-end ML project: data collection → embeddings → semantic search → classification → recommendations → REST API → multi-page frontend.

---


## Features

- **Semantic Search** - query by concept, not keyword. Powered by Sentence Transformers + FAISS
- **Topic Classifier** - predicts arXiv category (NLP, CV, ML, AI) using XGBoost on embeddings
- **Similar Papers** - content-based recommendation system using cosine similarity
- **AI Research Assistant** - paste a topic, get relevant papers + keyword extraction + category breakdown
- **Interactive Dashboard** - model metrics, t-SNE visualization, ablation study, error analysis
- **6-page Frontend** - Home, Discover, Assistant, Dashboard, Paper Detail, About

---

## ML Pipeline

```
arXiv API
    ↓
476 Research Papers (JSON)
    ↓
Sentence Transformer (all-MiniLM-L6-v2)
    ↓
Embeddings: 476 × 384 dimensions
    ↓
FAISS IndexFlatIP → Semantic Search
    ↓
XGBoost Classifier → Topic Prediction
```

---

## Experiments & Results

### Model Comparison

| Model | Weighted F1 |
|---|---|
| Logistic Regression | 0.6965 |
| Random Forest | 0.6830 |
| **XGBoost** | **0.7044** ✅ |

XGBoost won because embeddings create a high-dimensional space with non-linear structure. Tree-based boosting navigates this better than linear models.

### Embedding Model Comparison

| Model | F1 | Search Score | Speed |
|---|---|---|---|
| **all-MiniLM-L6-v2** | **0.6965** | **0.6049** | 16.5s |
| paraphrase-MiniLM-L6-v2 | 0.6859 | 0.6009 | 9.2s |

all-MiniLM-L6-v2 produces better representations for academic text. paraphrase-MiniLM-L6-v2 is 1.8x faster but sacrifices accuracy - a classic speed-accuracy tradeoff.

### Ablation Study — TF-IDF vs Sentence Transformers

| Representation | F1 | Dimensions |
|---|---|---|
| TF-IDF (sparse, lexical) | 0.6728 | 5000 |
| **Sentence Transformer (dense)** | **0.6965** | **384** |

Dense semantic embeddings outperform sparse lexical representations despite being 13x smaller. TF-IDF treats words independently - it can't know "neural network" and "deep learning" are related.

### Error Analysis

- 23 misclassifications out of 96 test samples (76% accuracy)
- Most common confusion: `stat.ML → cs.LG` (6 times)
- Insight: most errors occur between genuinely overlapping domains, suggesting category ambiguity in the data rather than pure model weakness

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (IndexFlatIP) |
| Classifier | XGBoost, Logistic Regression, Random Forest |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| Data Source | arXiv API |
| Visualization | Plotly (t-SNE), Chart.js |

---

## Project Structure

```
ScholarSync/
├── src/
│   ├── fetch_papers.py       # arXiv data collection
│   ├── embeddings.py         # sentence transformer + FAISS index
│   ├── classifier.py         # train LR, RF, XGBoost + evaluate
│   ├── compare_models.py     # embedding model comparison
│   ├── ablation.py           # TF-IDF vs embeddings ablation
│   ├── error_analysis.py     # misclassification analysis
│   ├── visualize.py          # t-SNE visualization
│   └── app.py                # FastAPI backend
├── frontend/
│   ├── index.html            # home + search
│   ├── discover.html         # browse + filter
│   ├── assistant.html        # AI research assistant
│   ├── dashboard.html        # metrics + visualizations
│   ├── paper.html            # paper detail + similar
│   ├── about.html            # architecture + analysis
│   ├── tsne.html             # interactive t-SNE plot
│   ├── styles.css            # shared styles
│   └── api.js                # shared API client
├── data/                     # generated files (not in git)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# clone
git clone https://github.com/ishhh115/ScholarSync.git
cd ScholarSync

# create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install -r requirements.txt

# fetch papers from arXiv
python src/fetch_papers.py

# build embeddings + FAISS index
python src/embeddings.py

# train classifiers
python src/classifier.py

# run the API
python -m uvicorn src.app:app --reload

# open frontend/index.html in your browser
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/search` | Semantic search over papers |
| GET | `/similar/{idx}` | Find similar papers by index |
| GET | `/stats` | Dataset statistics |

---

## Key Concepts Demonstrated

- **Embeddings** - converting text to dense vectors that capture semantic meaning
- **Cosine Similarity** - measuring similarity between vectors via angle, not magnitude
- **FAISS** - Facebook AI Similarity Search for efficient nearest-neighbor lookup
- **t-SNE** - dimensionality reduction to visualize 384D embeddings in 2D
- **XGBoost** - gradient boosted trees, handles non-linear feature interactions
- **Class Imbalance** - unequal category distribution affecting model performance
- **Ablation Study** - isolating the impact of a single component (TF-IDF vs embeddings)

---
