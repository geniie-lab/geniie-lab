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
            doc = self.docstore.get(docid)
            text = getattr(doc, 'text', None) or getattr(doc, 'body', None) or ''
            title = getattr(doc, 'title', None)
            return FullText(
                docid = docid,
                text = self.clean_text(text),
                title = self.clean_text(title) if title else None
            )
        except Exception as e:
            return Error(error_text=str(e))

    def generate_snippet(
        self,
        passage_chunks: List[str],
        query: str,
        max_segments: int = 1,
        max_segment_length: int = 150
    ) -> str:
        """
        Generate a snippet composed of the top N passages that match the
        query terms, based on lexical overlap.

        :param passage_chunks: List of passage texts (the text_chunks field).
        :param query: User query string.
        :param max_segments: Number of segments to return.
        :param max_segment_length: Max characters per segment.
        :return: Concatenated snippet of top-matching segments.
        """
        if not passage_chunks:
            return "No snippet available"

        query_terms = set(re.findall(r'\w+', query.lower()))

        # Score passages by number of query term overlaps
        def score_chunk(text: str) -> int:
            words = re.findall(r'\w+', text.lower())
            return sum(1 for w in words if w in query_terms)

        # Sort passages by overlap score (descending)
        ranked_chunks = sorted(passage_chunks, key=score_chunk, reverse=True)

        # Take top N and clean/truncate
        selected = [self.clean_text(raw)[:max_segment_length] for raw in ranked_chunks[:max_segments]]

        return " ... ".join(selected) if selected else "No snippet available"

    def search_index_with_snippets(
        self,
        query: str,
        start: int = 0,
        size: int = 10
    ) -> Serp:
        query_vector = self.model.encode_query(query).tolist()
        search_body = {
            "from": start,
            "size": size,
            # Documents are chunked at indexing time; per-passage dense
            # vectors live in the nested text_chunks_embedding field and the
            # passage texts in the parallel top-level text_chunks array. The
            # query is encoded client-side (participants' OpenSearch users
            # have no ML-plugin permissions), so a raw vector is sent.
            "query": {
                "nested": {
                    "path": "text_chunks_embedding",
                    "score_mode": "max",  # best-matching passage sets the doc score
                    "query": {
                        "knn": {
                            "text_chunks_embedding.knn": {
                                "vector": query_vector,
                                "k": size
                            }
                        }
                    }
                }
            },
            "_source": {
                "includes": ["docid", "title", "text_chunks"]
            }
        }
        response = self.client.search(index=self.index_name, body=search_body)
        total_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        if total_hits == 0:
            return Serp(hits=0, results=[])
        items: List[SearchResultItem] = []
        for idx, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            src = hit.get("_source", {})
            passage_chunks = src.get("text_chunks", [])

            snippet_text = self.generate_snippet(passage_chunks, query=query)

            items.append(SearchResultItem(
                ranking=start + idx,
                docid=src.get("docid"),
                title=self.clean_text(src.get("title", "No Title")),
                snippet=snippet_text,
                score=hit.get("_score") or 0.0
            ))

        return Serp(hits=total_hits, results=items)