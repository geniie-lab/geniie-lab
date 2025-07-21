# Standard library
from dataclasses import dataclass
from typing import List, Optional

# Third-party libraries
from dataclasses_json import DataClassJsonMixin, dataclass_json

@dataclass_json
@dataclass
class SearchResultItem(DataClassJsonMixin):
    ranking: int
    docid: str
    title: str
    snippet: str

@dataclass_json
@dataclass
class Serp(DataClassJsonMixin):
    hits: int
    results: List[SearchResultItem]

@dataclass
class FullText:
    docid: str
    text: str
    title: Optional[str] = None