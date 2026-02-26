# Standard library
import re
from typing import List, Union, Optional

# Third-party libraries
import ir_datasets
from opensearchpy import OpenSearch
import torch
from sentence_transformers.sparse_encoder import SparseEncoder

# Local application imports
from geniie_lab.dataclasses.serp import FullText, SearchResultItem, Serp
from geniie_lab.dataclasses.setting import Error

class OpenSearchClientSplade:
    """
    Encapsulates OpenSearch client operations, including fetching full documents
    and searching with highlighted snippets.
    """
    def __init__(
        self,
        index_name: str,
        dataset_name: str,
        encode_model: str,
        host: str = "localhost",
        port: int = 9200,
        http_auth: Optional[tuple[str, str]] = None,
        use_ssl: bool = True
    ):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self.index_name = index_name
        self.dataset = ir_datasets.load(dataset_name)

        self.encode_model = encode_model
        model_name = self.encode_model or "naver/splade-cocondenser-ensembledistil"
        self.model = SparseEncoder(model_name, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model.cuda()

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return " ".join(text.splitlines())

    def fetch_fulltext(self, docid: str) -> Union[FullText, Error]:
        try:
            self.docstore = self.dataset.docs_store()
            text = self.docstore.get(docid).text 
            return FullText(
                docid = docid,
                text = self.clean_text(text)
            )
        except Exception as e:
            return Error(error_text=str(e))
        
    def search_index_with_snippets(
        self,
        query: str,
        start: int = 0,
        size: int = 10
    ) -> Serp:
        query_tensor = self.model.encode_query([query])
        query_embedding = self.model.decode(query_tensor)
        search_body = {
            "from": start,
            "size": size,
            "query": {
                "neural_sparse": {
                    "sparse_embedding": {
                        "query_tokens": dict(query_embedding[0])
                    }
                }
            },
            "highlight": {
                "fields": {
                    "text": {
                        "type": "plain",
                        "fragment_size": 150,
                        "number_of_fragments": 1
                    }
                }
            }
        }
        response = self.client.search(index=self.index_name, body=search_body)
        total_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        if total_hits == 0:
            return Serp(hits=0, results=[])
        items: List[SearchResultItem] = []
        for idx, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            src = hit.get("_source", {})
            raw_snippets = hit.get("highlight", {}).get("text", [src.get("text", "")[:150]])
            snippet_text = " ... ".join(self.clean_text(s) for s in raw_snippets)
            items.append(SearchResultItem(
                ranking=start + idx,
                docid=src.get("docid"),
                title=self.clean_text(src.get("title", "No Title")),
                snippet=snippet_text
            ))
        return Serp(hits=total_hits, results=items)