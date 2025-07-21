# Standard library
import re
from typing import List, Union, Optional

# Third-party libraries
import ir_datasets
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Local application imports
from geniie_lab.dataclasses.serp import FullText, SearchResultItem, Serp
from geniie_lab.dataclasses.setting import Error

class OpenSearchClientDPR:
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

        # Load model inside the function (only once)
        self.encode_model = encode_model
        model_name = self.encode_model or "sentence-transformers/msmarco-distilbert-base-tas-b"
        self.model = SentenceTransformer(model_name)

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

    def generate_snippet(
        self,
        passage_chunks: List[dict],
        query: str,
        max_segments: int = 1,
        max_segment_length: int = 150
    ) -> str:
        """
        Generate a snippet composed of the top N segments (passage chunks)
        that match the query terms, based on lexical overlap.

        :param passage_chunks: List of passage chunk dicts.
        :param query: User query string.
        :param max_segments: Number of segments to return.
        :param max_segment_length: Max characters per segment.
        :return: Concatenated snippet of top-matching segments.
        """
        if not passage_chunks:
            return "No snippet available"

        query_terms = set(re.findall(r'\w+', query.lower()))

        # Score chunks by number of query term overlaps
        def score_chunk(chunk):
            text = chunk.get("text", "")
            words = re.findall(r'\w+', text.lower())
            return sum(1 for w in words if w in query_terms)

        # Sort chunks by overlap score (descending)
        ranked_chunks = sorted(passage_chunks, key=score_chunk, reverse=True)

        # Take top N and clean/truncate
        selected = []
        for chunk in ranked_chunks[:max_segments]:
            raw = chunk.get("text", "")
            clean = self.clean_text(raw)[:max_segment_length]
            selected.append(clean)

        return " ... ".join(selected) if selected else "No snippet available"

    def search_index_with_snippets(
        self,
        query: str,
        start: int = 0,
        size: int = 10
    ) -> Serp:
        query_vector = self.model.encode(query).tolist()
        search_body = {
            "from": start,
            "size": size,
            "query": {
                "nested": {
                    "path": "passage_chunk",
                    "score_mode": "max",
                    "query": {
                        "knn": {
                            "passage_chunk.embedding": {
                                "vector": query_vector,
                                "k": size
                            }
                        }
                    }
                }
            },
            "_source": {
                "includes": ["docid", "title", "passage_chunk"],
                "excludes": ["passage_chunk.embedding"]
            }
        }
        response = self.client.search(index=self.index_name, body=search_body)
        total_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        if total_hits == 0:
            return Serp(hits=0, results=[])
        items: List[SearchResultItem] = []
        for idx, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            src = hit.get("_source", {})
            passage_chunks = src.get("passage_chunk", [])

            snippet_text = self.generate_snippet(passage_chunks, query=query)

            items.append(SearchResultItem(
                ranking=start + idx,
                docid=src.get("docid"),
                title=self.clean_text(src.get("title", "No Title")),
                snippet=snippet_text
            ))

        return Serp(hits=total_hits, results=items)