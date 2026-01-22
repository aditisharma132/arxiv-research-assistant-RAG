# ingest_papers.py (v2 - metadata-only)
import arxiv
import json
import os
import time

def fetch_arxiv_metadata(query="cat:cs.CL OR cat:cs.LG", max_results=100, output_dir="papers_metadata"):
    os.makedirs(output_dir, exist_ok=True)
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    saved = 0
    for result in search.results():
        try:
            paper_id = result.get_short_id()
            data = {
                "paper_id": paper_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": [str(a) for a in result.authors],
                "published": str(result.published),
                "pdf_url": result.pdf_url
            }
            with open(f"{output_dir}/{paper_id}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved metadata: {paper_id}")
            saved += 1
            time.sleep(3)
            if saved >= max_results:
                break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    fetch_arxiv_metadata(max_results=100)