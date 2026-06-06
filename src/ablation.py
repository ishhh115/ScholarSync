import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def load_papers():
    with open("data/papers.json", "r") as f:
        return json.load(f)

def simplify_labels(papers):
    main_categories = {"cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"}
    return [p["category"] if p["category"] in main_categories else "other" for p in papers]

def evaluate_tfidf(papers, labels):
    print("Evaluating: TF-IDF + Logistic Regression")
    abstracts = [p["abstract"] for p in papers]
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(abstracts).toarray()
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"  TF-IDF F1: {f1:.4f}")
    print(f"  Dimensions: {X.shape[1]}")
    return round(f1, 4), X.shape[1]

def evaluate_embeddings(labels):
    print("Evaluating: Sentence Transformer Embeddings + Logistic Regression")
    embeddings = np.load("data/embeddings.npy")
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"  Embeddings F1: {f1:.4f}")
    print(f"  Dimensions: {embeddings.shape[1]}")
    return round(f1, 4), embeddings.shape[1]

if __name__ == "__main__":
    papers = load_papers()
    labels = simplify_labels(papers)
    
    tfidf_f1, tfidf_dims = evaluate_tfidf(papers, labels)
    embed_f1, embed_dims = evaluate_embeddings(labels)
    
    print("\n=== ABLATION STUDY RESULTS ===")
    print(f"{'Representation':<35} {'F1':>8} {'Dims':>8}")
    print("-" * 55)
    print(f"{'TF-IDF (sparse, lexical)':<35} {tfidf_f1:>8} {tfidf_dims:>8}")
    print(f"{'Sentence Transformer (dense)':<35} {embed_f1:>8} {embed_dims:>8}")
    print(f"\nImprovement: +{round(embed_f1 - tfidf_f1, 4)} F1 with dense embeddings")
    
    results = {
        "tfidf": {"f1": tfidf_f1, "dims": tfidf_dims, "type": "sparse lexical"},
        "embeddings": {"f1": embed_f1, "dims": embed_dims, "type": "dense semantic"}
    }
    
    with open("data/ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to data/ablation.json")