import json
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def load_data():
    embeddings = np.load("data/embeddings.npy")
    with open("data/papers_metadata.pkl", "rb") as f:
        papers = pickle.load(f)
    main_categories = {"cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"}
    labels = [p["category"] if p["category"] in main_categories else "other" for p in papers]
    return embeddings, labels, papers

if __name__ == "__main__":
    embeddings, labels, papers = load_data()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        embeddings, y, list(range(len(papers))), test_size=0.2, random_state=42
    )

    clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # find mistakes
    mistakes = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            paper_idx = idx_test[i]
            mistakes.append({
                "title": papers[paper_idx]["title"],
                "true_label": le.inverse_transform([true])[0],
                "predicted_label": le.inverse_transform([pred])[0],
                "abstract": papers[paper_idx]["abstract"][:200]
            })

    print(f"Total mistakes: {len(mistakes)} / {len(y_test)}")
    print(f"Accuracy: {1 - len(mistakes)/len(y_test):.4f}\n")

    # group by confusion pair
    confusion_pairs = {}
    for m in mistakes:
        pair = f"{m['true_label']} → {m['predicted_label']}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    print("Most common confusion pairs:")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {count}")

    print("\nExample mistakes:")
    for m in mistakes[:5]:
        print(f"\n  Title: {m['title']}")
        print(f"  True: {m['true_label']} | Predicted: {m['predicted_label']}")

    with open("data/error_analysis.json", "w") as f:
        json.dump({
            "total_mistakes": len(mistakes),
            "total_test": len(y_test),
            "accuracy": round(1 - len(mistakes)/len(y_test), 4),
            "confusion_pairs": confusion_pairs,
            "examples": mistakes[:8]
        }, f, indent=2)

    print("\nSaved to data/error_analysis.json")