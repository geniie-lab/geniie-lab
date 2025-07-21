from dataclasses import dataclass

@dataclass
class Qrel:
    query_id: str
    doc_id: str
    relevance: int

@dataclass
class Qrels:
    items: list[Qrel]

    def __init__(self):
        self.items = []
        self.index = {}

    def add(self, query_id: str, doc_id: str, relevance: int):
        qrel = Qrel(query_id, doc_id, relevance)
        self.items.append(Qrel(query_id, doc_id, relevance))
        self.index[(query_id, doc_id)] = relevance

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def get(self, query_id: str, doc_id: str, default=None):
        return self.index.get((query_id, doc_id), default)

@dataclass
class RunItem:
    query_id: str
    doc_id: str
    score: float

@dataclass
class Run:
    items: list[RunItem]

    def __init__(self):
        self.items = []

    def add(self, query_id: str, doc_id: str, score: float):
        self.items.append(RunItem(query_id, doc_id, score))

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)