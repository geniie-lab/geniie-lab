# Standard library
from dataclasses import dataclass, field
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
    # Retrieval score from the search engine, used for evaluation only.
    # repr=False keeps it out of the SERP rendered into LLM instructions:
    # the simulated searcher must not see engine scores.
    score: float = field(default=0.0, repr=False)

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