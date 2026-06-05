import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import time

def load_papers():
    with open("data/papers.json", "r") as f:
        return json.load(f)

def simplify_labels(papers):
    main_categories = {"cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"}
    return ["other" if p["category"] not in main_categories else p["category"] for p in papers]

def evaluate_model(model_name, papers, labels):
    print(f"\nEvaluating: {model_name}")
    
    model = SentenceTransformer(model_name)
    abstracts = [p["abstract"] for p in papers]
    
    start = time.time()
    embeddings = model.encode(abstracts, show_progress_bar=True)
    embed_time = round(time.time() - start, 2)
    
    # classifier F1
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # search quality — test with 5 queries, measure avg top score
    test_queries = [
        "attention mechanism transformer",
        "object detection convolutional neural network",
        "natural language processing BERT",
        "reinforcement learning policy gradient",
        "generative adversarial network image synthesis"
    ]
    
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    normed = embeddings.copy()
    faiss.normalize_L2(normed)
    index.add(normed)
    
    avg_top_score = 0
    for q in test_queries:
        qe = model.encode([q])
        faiss.normalize_L2(qe)
        scores, _ = index.search(qe, 1)
        avg_top_score += scores[0][0]
    avg_top_score /= len(test_queries)
    
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Avg Top Search Score: {avg_top_score:.4f}")
    print(f"  Embedding Time: {embed_time}s")
    print(f"  Dimensions: {embeddings.shape[1]}")
    
    return {
        "model": model_name,
        "f1": round(f1, 4),
        "avg_search_score": round(float(avg_top_score), 4),
        "embed_time_sec": embed_time,
        "dimensions": embeddings.shape[1]
    }

if __name__ == "__main__":
    papers = load_papers()
    labels = simplify_labels(papers)
    
    models = [
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2"
    ]
    
    results = []
    for m in models:
        r = evaluate_model(m, papers, labels)
        results.append(r)
    
    print("\n\n=== COMPARISON RESULTS ===")
    print(f"{'Model':<35} {'F1':>8} {'Search Score':>14} {'Time(s)':>10} {'Dims':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<35} {r['f1']:>8} {r['avg_search_score']:>14} {r['embed_time_sec']:>10} {r['dimensions']:>6}")
    
    with open("data/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nSaved to data/model_comparison.json")