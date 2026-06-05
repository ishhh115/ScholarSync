from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

app = FastAPI(title="ScholarSync API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load everything on startup
print("Loading model and index...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/papers.index")

with open("data/papers_metadata.pkl", "rb") as f:
    papers = pickle.load(f)

with open("data/classifier.pkl", "rb") as f:
    classifier_data = pickle.load(f)
    classifier = classifier_data["model"]
    label_encoder = classifier_data["label_encoder"]

print("Ready!")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "ScholarSync API is running"}

@app.post("/search")
def search(request: SearchRequest):
    # embed the query
    query_embedding = model.encode([request.query])
    faiss.normalize_L2(query_embedding)
    
    # search
    scores, indices = index.search(query_embedding, request.top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        paper = papers[idx]
        
        # predict category
        paper_embedding = np.load("data/embeddings.npy")[idx].reshape(1, -1)
        predicted_category = label_encoder.inverse_transform(
            classifier.predict(paper_embedding)
        )[0]
        
        results.append({
            "title": paper["title"],
            "abstract": paper["abstract"][:400] + "...",
            "category": paper["category"],
            "predicted_category": predicted_category,
            "similarity_score": round(float(score), 4)
        })
    
    return {"query": request.query, "results": results}

@app.get("/stats")
def stats():
    return {
        "total_papers": len(papers),
        "index_size": index.ntotal,
    }