import json
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    embeddings = np.load("data/embeddings.npy")
    with open("data/papers_metadata.pkl", "rb") as f:
        papers = pickle.load(f)
    
    # simplify categories — keep only main ones, group rest as "other"
    main_categories = {"cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"}
    labels = []
    for p in papers:
        cat = p["category"]
        if cat in main_categories:
            labels.append(cat)
        else:
            labels.append("other")
    
    return embeddings, labels

def train_and_evaluate():
    print("Loading embeddings and labels...")
    X, labels = load_data()
    
    # encode string labels to numbers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"Classes: {le.classes_}")
    print(f"Total samples: {len(X)}")
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
}
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        results[name] = {"model": model, "f1": f1, "y_pred": y_pred}
        print(f"{name} F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # pick best model
    best_name = max(results, key=lambda x: results[x]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} with F1: {results[best_name]['f1']:.4f}")
    
    # plot confusion matrix for best model
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("data/confusion_matrix.png")
    print("Confusion matrix saved to data/confusion_matrix.png")
    
    # save best model
    with open("data/classifier.pkl", "wb") as f:
        pickle.dump({"model": best_model, "label_encoder": le}, f)
    print("Model saved to data/classifier.pkl")

if __name__ == "__main__":
    train_and_evaluate()