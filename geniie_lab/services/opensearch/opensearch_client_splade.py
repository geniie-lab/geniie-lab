# Standard library
import re
from typing import List, Union, Optional

# Third-party libraries
import ir_datasets
from opensearchpy import OpenSearch
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

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
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    @torch.no_grad()
    def splade_encode_to_bow(self, text, tokenizer, model, max_doc_length=512, top_k=30):
        """
        Encode input text using SPLADE and return a weighted BoW string.

        :param text: input string (document or chunk)
        :param tokenizer: HuggingFace tokenizer
        :param model: SPLADE model (masked LM)
        :param max_doc_length: max input tokens (SPLADE supports long input)
        :param top_k: number of top tokens to keep by weight
        :return: string of repeated tokens (weighted BoW)
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_doc_length)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)  # [seq_len, vocab_size]

        # Apply log1p(ReLU(x)) to each token dimension (SPLADE's sparse activation trick)
        sparse_weights = torch.log1p(F.relu(logits)).max(dim=0).values  # [vocab_size]

        # Get top-k weighted vocab terms
        topk = torch.topk(sparse_weights, k=top_k)
        token_ids = topk.indices.tolist()
        weights = topk.values.tolist()

        # Build weighted term list: repeat each token according to its weight
        bow_tokens = []
        for token_id, weight in zip(token_ids, weights):
            token = tokenizer.convert_ids_to_tokens(token_id)
            repeat_count = max(1, int(round(weight)))
            bow_tokens.extend([token] * repeat_count)

        return " ".join(bow_tokens)

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
        bow_query = self.splade_encode_to_bow(query, self.tokenizer, self.model)
        search_body = {
            "from": start,
            "size": size,
            "query": {
                "match": {
                    "splade_text": {
                        "query": bow_query
                    }
                }
            },
            "highlight": {
                "fields": {
                    "splade_text": {
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
            raw_snippets = hit.get("highlight", {}).get("splade_text", [src.get("text", "")[:150]])
            snippet_text = " ... ".join(self.clean_text(s) for s in raw_snippets)
            items.append(SearchResultItem(
                ranking=start + idx,
                docid=src.get("docid"),
                title=self.clean_text(src.get("title", "No Title")),
                snippet=snippet_text
            ))
        return Serp(hits=total_hits, results=items)