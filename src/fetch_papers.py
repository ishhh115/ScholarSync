import requests
import json
import time
import xml.etree.ElementTree as ET

def fetch_papers(query, max_results=100):
    base_url = "http://export.arxiv.org/api/query"
    papers = []
    
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text[:200]}")
        return []
    
    if not response.text.strip():
        print("Empty response from arXiv")
        return []

    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        print(f"Response was: {response.text[:300]}")
        return []
    
    entries = root.findall("atom:entry", namespace)
    
    for entry in entries:
        try:
            title = entry.find("atom:title", namespace).text.strip()
            abstract = entry.find("atom:summary", namespace).text.strip()
            paper_id = entry.find("atom:id", namespace).text.strip()
            
            category_el = entry.find("arxiv:primary_category", namespace)
            category = category_el.attrib.get("term", "unknown") if category_el is not None else "unknown"
            
            papers.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "category": category,
                "query": query
            })
        except Exception as e:
            continue
    
    return papers

if __name__ == "__main__":
    queries = ["machine learning", "natural language processing", "computer vision", "reinforcement learning", "deep learning"]
    all_papers = []
    
    for query in queries:
        print(f"Fetching papers for: {query}")
        papers = fetch_papers(query, max_results=100)
        all_papers.extend(papers)
        print(f"Got {len(papers)} papers")
        time.sleep(5)
    
    seen = set()
    unique_papers = []
    for p in all_papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique_papers.append(p)
    
    with open("data/papers.json", "w") as f:
        json.dump(unique_papers, f, indent=2)
    
    print(f"\nTotal unique papers saved: {len(unique_papers)}")