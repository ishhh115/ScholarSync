import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_papers():
    with open("data/papers.json", "r") as f:
        return json.load(f)

def build_embeddings():
    print("Loading papers...")
    papers = load_papers()
    print(f"Loaded {len(papers)} papers")
    
    # load the model
    print("Loading sentence transformer model (first time downloads ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # embed each paper's abstract
    print("Creating embeddings...")
    abstracts = [p["abstract"] for p in papers]
    embeddings = model.encode(abstracts, show_progress_bar=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # build FAISS index for fast similarity search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = inner product (cosine similarity)
    
    # normalize so inner product = cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    
    # save everything
    faiss.write_index(index, "data/papers.index")
    np.save("data/embeddings.npy", embeddings)
    with open("data/papers_metadata.pkl", "wb") as f:
        pickle.dump(papers, f)
    
    print("Saved index, embeddings, and metadata")
    return model, index, papers, embeddings

def search(query, model, index, papers, top_k=5):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        paper = papers[idx]
        results.append({
            "title": paper["title"],
            "abstract": paper["abstract"][:300] + "...",
            "category": paper["category"],
            "score": float(score)
        })
    return results

if __name__ == "__main__":
    model, index, papers, embeddings = build_embeddings()
    
    # test search
    print("\nTesting search...")
    query = "attention mechanism in neural networks"
    results = search(query, model, index, papers)
    
    print(f"\nTop 5 results for: '{query}'")
    for i, r in enumerate(results):
        print(f"\n{i+1}. {r['title']}")
        print(f"   Category: {r['category']}")
        print(f"   Score: {r['score']:.4f}")