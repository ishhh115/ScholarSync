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

@app.get("/similar/{paper_index}")
def similar_papers(paper_index: int, top_k: int = 5):
    if paper_index < 0 or paper_index >= len(papers):
        return {"error": "Invalid paper index"}
    
    # load embeddings
    embeddings = np.load("data/embeddings.npy")
    
    # use the paper's own embedding as the query
    paper_embedding = embeddings[paper_index].reshape(1, -1)
    faiss.normalize_L2(paper_embedding)
    
    # search — top_k+1 because the paper itself will be the first result
    scores, indices = index.search(paper_embedding, top_k + 1)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == paper_index:
            continue  # skip the paper itself
        paper = papers[idx]
        results.append({
            "index": int(idx),
            "title": paper["title"],
            "category": paper["category"],
            "similarity_score": round(float(score), 4)
        })
    
    return {
        "source_paper": papers[paper_index]["title"],
        "similar_papers": results
    }

@app.get("/stats")
def stats():
    return {
        "total_papers": len(papers),
        "index_size": index.ntotal,
    }