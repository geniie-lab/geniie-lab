import os
from geniie_lab.dataclasses.description import ToolDescription
from geniie_lab.dataclasses.setting import ExperimentSettings
from geniie_lab.services.opensearch.opensearch_client_bm25 import OpenSearchClientBM25
from geniie_lab.services.opensearch.opensearch_client_dpr import OpenSearchClientDPR
from geniie_lab.services.opensearch.opensearch_client_protocol import OpenSearchClientProtocol
from geniie_lab.services.opensearch.opensearch_client_splade import OpenSearchClientSplade

class OpenSearchClientFactory:
    def create_opensearch_client(self, settings: ExperimentSettings, tool: ToolDescription) -> OpenSearchClientProtocol:

        http_auth = (
            os.environ.get("OPENSEARCH_ADMIN_USER", "admin"),
            os.environ.get("OPENSEARCH_ADMIN_PASS", "admin"),
        )

        if tool.ranking_model == "bm25":
            return OpenSearchClientBM25(
                index_name=tool.index_name,
                host=tool.host,
                port=tool.port,
                dataset_name = settings.topicset.name,
                http_auth=http_auth
            )
        elif tool.ranking_model == "splade":
            return OpenSearchClientSplade(
                index_name=tool.index_name,
                host=tool.host,
                port=tool.port,
                dataset_name = settings.topicset.name,
                http_auth=http_auth,
                encode_model=tool.encode_model
            )
        elif tool.ranking_model == "dpr":
            return OpenSearchClientDPR(
                index_name=tool.index_name,
                host=tool.host,
                port=tool.port,
                dataset_name = settings.topicset.name,
                http_auth=http_auth,
                encode_model=tool.encode_model
            )
        else:
            raise ValueError(f"Unknown ranking_model: {tool.ranking_model}")