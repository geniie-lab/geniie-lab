import os
import ir_datasets
from ir_datasets.datasets.base import Dataset, YamlDocumentation
from ir_datasets.formats import TrecQrels, JsonlDocs, JsonlQueries
from ir_datasets.util import LocalDownload, Cache

NAME = 'custom-amazon-esci'

LABEL_MAP = {"E": 3, "S": 2, "C": 1, "I": 0}
QREL_DEFS = {
    3: 'exact',
    2: 'substitute',
    1: 'complement',
    0: 'irrelevant'
}

def _init():
    base_path = ir_datasets.util.home_path() / NAME

    # Dataset file paths (adapt these to actual local paths if needed)
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DOCS_PATH = os.path.join(ROOT, 'products.jsonl')

    # Split-specific paths
    SPLITS = {
        'train': {
            'queries': os.path.join(ROOT, 'train_queries.jsonl'),
            'qrels': os.path.join(ROOT, 'train_qrels.tsv'),
        },
        'test': {
            'queries': os.path.join(ROOT, 'test_queries.jsonl'),
            'qrels': os.path.join(ROOT, 'test_qrels.tsv'),
        }
    }

    # LocalDownload wrappers
    DL_DOCS = LocalDownload(DOCS_PATH)
    DL_QUERIES = {split: LocalDownload(paths['queries']) for split, paths in SPLITS.items()}
    DL_QRELS = {split: LocalDownload(paths['qrels']) for split, paths in SPLITS.items()}

    subsets = {}

    # Shared docs
    docs = JsonlDocs(ir_datasets.util.Cache(DOCS_PATH, base_path / 'products.jsonl'))

    # Create Dataset objects for each split
    for split in SPLITS:
        queries = JsonlQueries(Cache(DL_QUERIES[split], base_path / f"{split}_queries.jsonl"))
        qrels = TrecQrels(Cache(DL_QRELS[split], base_path / f"{split}_qrels.tsv"), QREL_DEFS)
        ds = Dataset(docs, queries, qrels)
        ir_datasets.registry.register(f'{NAME}/{split}', ds)
        subsets[split] = ds

    # Documentation for root dataset
    documentation = YamlDocumentation(f'{NAME}.yaml')
    base = Dataset(documentation('_'))
    ir_datasets.registry.register(f'{NAME}', base)

    return base, subsets

base, subsets = _init()