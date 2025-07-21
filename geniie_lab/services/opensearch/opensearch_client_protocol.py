from typing import Protocol, Union
from geniie_lab.dataclasses.serp import Serp, FullText
from geniie_lab.dataclasses.setting import Error

class OpenSearchClientProtocol(Protocol):
    def clean_text(self, text: str) -> str: ...

    def search_index_with_snippets(self, query: str, start: int, size: int) -> Serp: ...

    def fetch_fulltext(self, docid: str) -> Union[FullText, Error]: ...