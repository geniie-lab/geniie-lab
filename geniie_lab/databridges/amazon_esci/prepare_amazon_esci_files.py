from datasets import load_dataset
import json
from collections import defaultdict

# === Load full product metadata once ===
products = load_dataset("milistu/amazon-esci-data", name="products", split="train")
products_by_id = {
    p["product_id"]: {
        "doc_id": p["product_id"],
        "title": p["product_title"],
        "text": p.get("product_description", "")
    } for p in products
}

# === Prepare files ===
for split in ["train", "test"]:
    queries = load_dataset("milistu/amazon-esci-data", name="queries", split=split)

    # For products.jsonl — write once per run
    if split == "train":
        seen_docs = set()
        with open("products.jsonl", "w", encoding="utf-8") as f:
            for q in queries:
                pid = q["product_id"]
                if pid not in seen_docs and pid in products_by_id:
                    json.dump(products_by_id[pid], f, ensure_ascii=False)
                    f.write("\n")
                    seen_docs.add(pid)

    # For queries
    with open(f"{split}_queries.jsonl", "w", encoding="utf-8") as fq, \
         open(f"{split}_qrels.tsv", "w", encoding="utf-8") as fr:
        
        seen_qids = set()
        for q in queries:
            # Write query record
            if q["query_id"] not in seen_qids:
                query_obj = {
                    "query_id": q["query_id"],
                    "text": q["query"]
                }
                json.dump(query_obj, fq, ensure_ascii=False)
                fq.write("\n")
                seen_qids.add(q["query_id"])
            
            # Write qrel line
            label_map = {"E": 3, "S": 2, "C": 1, "I": 0}
            relevance = label_map.get(q["esci_label"], 0)
            fr.write(f"{q['query_id']}\tQ0\t{q['product_id']}\t{relevance}\n")
