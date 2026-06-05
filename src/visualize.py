import json
import numpy as np
import pickle
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio

def load_data():
    embeddings = np.load("data/embeddings.npy")
    with open("data/papers_metadata.pkl", "rb") as f:
        papers = pickle.load(f)
    
    main_categories = {"cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"}
    labels = [p["category"] if p["category"] in main_categories else "other" for p in papers]
    titles = [p["title"][:80] + "..." if len(p["title"]) > 80 else p["title"] for p in papers]
    
    return embeddings, labels, titles

if __name__ == "__main__":
    print("Loading embeddings...")
    embeddings, labels, titles = load_data()
    
    print("Running t-SNE (this takes ~30 seconds)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print("Building interactive plot...")
    import pandas as pd
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "category": labels,
        "title": titles
    })
    
    color_map = {
        "cs.LG": "#00e5ff",
        "cs.CL": "#7b61ff",
        "cs.CV": "#00d68f",
        "cs.AI": "#ffb347",
        "stat.ML": "#ff6b6b",
        "other": "#5a6a7a"
    }
    
    fig = px.scatter(
        df, x="x", y="y",
        color="category",
        hover_data={"title": True, "x": False, "y": False},
        color_discrete_map=color_map,
        title="t-SNE Visualization of Paper Embeddings",
        labels={"category": "Category"}
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        plot_bgcolor="#0d1520",
        paper_bgcolor="#0e1318",
        font=dict(family="DM Mono", color="#e8edf2"),
        title=dict(font=dict(size=16, color="#e8edf2")),
        legend=dict(bgcolor="#0e1318", bordercolor="#1e2a35", borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    # save as standalone HTML
    pio.write_html(fig, "frontend/tsne.html", include_plotlyjs=True, full_html=True)
    print("Saved to frontend/tsne.html")
    
    # also save the 2D coords for embedding in dashboard
    np.save("data/tsne_coords.npy", embeddings_2d)
    with open("data/tsne_labels.json", "w") as f:
        json.dump({"labels": labels, "titles": titles, "x": embeddings_2d[:,0].tolist(), "y": embeddings_2d[:,1].tolist()}, f)
    print("Saved coords to data/tsne_labels.json")