import re
import sys
import pprint
from dataclasses import dataclass
import ir_datasets
from typing import Protocol, Dict, Type
from itertools import islice

from geniie_lab.dataclasses.setting import ExperimentSettings, ExperimentState, StageConfig, Error
from geniie_lab.dataclasses.description import ModelDescription, ToolDescription
from geniie_lab.dataclasses.topic import (
    TitleDescriptionNarrativeTopic, FullTopic,
    TitleDescriptionTopic,
    TitleNarrativeTopic,
    TitleOnlyTopic,
    TopicList,
    BaseTopic
)
from geniie_lab.dataclasses.instruction import (
    ClickInstruction,
    QueryFormulationInstruction,
    QueryReFormulationInstruction,
    RelevanceJudgementInstruction,
    NextActionInstruction,
)
from geniie_lab.dataclasses.output import (
    ClickExperimentOutput,
    QueryExperimentOutput,
    QueryReformulationExperimentOutput,
    RankingExperimentOutput,
    RelevanceJudgementExperimentOutput,
    NextActionOutput,
)
from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Action, NextAction
from geniie_lab.services.llm.llm_service_factory import LLMServiceFactory
from geniie_lab.services.llm.llm_service_protocol import LLMServiceProtocol
from geniie_lab.services.measure_service import MeasureService, Qrels, Run
from geniie_lab.services.opensearch.opensearch_client_factory import OpenSearchClientFactory
from geniie_lab.services.opensearch.opensearch_client_protocol import OpenSearchClientProtocol

class ExperimentStage(Protocol):
    def __init__(self, config: StageConfig): ...
    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState: ...
    
class QueryFormulationStage:
    DEFAULT_INSTRUCTION = """
        Review the provided descriptions of task, corpus, tool and search topic. Formulate a search query.
    """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        print("\n--- Running: Query Formulation Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        qf_instruction = QueryFormulationInstruction(instruction=instruction_text, task=settings.task, corpus=settings.corpus, tool=tool, topic=state.topic)

        state.query = llm_service.create_query(model.name, model.temperature, model.top_p, state.memory, qf_instruction)

        output = QueryExperimentOutput(
            session_name = settings.name,
            model = model.name,
            task = settings.task.name,
            dataset = settings.topicset.name,
            topic_id = state.topic.id,
            query = state.query.query,
            start = settings.task.start_offset,
            size = settings.task.serp_size
        )
        print(output.to_json(ensure_ascii=False))
        return state

class RankingStage:
    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        print("\n--- Running: Ranking Stage ---", file=sys.stderr)

        query_text = getattr(state.query, "query", None)
        start_offset = getattr(state.query, "start", 0)

        state.serp = opensearch_client.search_index_with_snippets(query_text, start=start_offset, size=settings.task.serp_size)
        state.docids = [item.docid for item in state.serp.results] if state.serp and state.serp.results else []

        dataset = ir_datasets.load(settings.topicset.name)
        if callable(dataset):
            dataset = dataset()

        qrels = Qrels()
        for row in dataset.qrels_iter():
            if row.query_id == state.topic.id:
                qrels.add(row.query_id, row.doc_id, row.relevance)

        run = Run()
        for result in state.serp.results:
            run.add(state.topic.id, result.docid, result.ranking)
        results = MeasureService().calc(settings.task.measurement, qrels, run)

        output = RankingExperimentOutput(
            session_name=settings.name,
            model=model.name,
            ranker=tool.ranking_model,
            task=settings.task.name,
            dataset=settings.topicset.name,
            topic_id=state.topic.id,
            doc_ids=state.docids,
            start = settings.task.start_offset,
            size=settings.task.serp_size,
            performance=results
        )
        print(output.to_json(ensure_ascii=False))
        return state
    
class ClickStage:
    DEFAULT_INSTRUCTION = """
            Review the search topic, submitted query, and retrieved search results. Then, select a set of documents that are likely to contain relevant information to the search topic. Return an empty list if none of the results appears relevant.
        """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        if not state.serp:
            state.error = "SERP not found, cannot run ClickStage."
            return state

        print("\n--- Running: Click Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        click_instruction = ClickInstruction(instruction=instruction_text, serp=state.serp)

        state.clicks = llm_service.create_clicks(model.name, model.temperature, model.top_p, state.memory, click_instruction)

        output = ClickExperimentOutput(
            session_name=settings.name,
            model=model.name,
            task=settings.task.name,
            dataset=settings.topicset.name,
            topic_id=state.topic.id,
            rankings=state.clicks.ranking_list
        )
        print(output.to_json(ensure_ascii=False))

        return state

class RelevanceJudgementStage:
    DEFAULT_INSTRUCTION = """
            Evaluate the relevance of the document based on the topic description, submitted query, and the full text provided.
            Your response should include:
            - `label` (required): Indicate whether the document is `Relevant` or `NotRelevant`.
        """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        if not state.clicks or not state.serp or not state.clicks.ranking_list:
            state.error = "Clicks/SERP not found or no documents clicked, cannot run RelevanceJudgementStage."
            return state

        dataset = ir_datasets.load(settings.topicset.name)
        if callable(dataset):
            dataset = dataset()

        qrels = Qrels()
        for row in dataset.qrels_iter():
            if row.query_id == state.topic.id:
                qrels.add(row.query_id, row.doc_id, row.relevance)
                
        print("\n--- Running: Relevance Judgement Stage ---", file=sys.stderr)
        for click_index in state.clicks.ranking_list:
            if click_index < 1 or click_index > len(state.serp.results):
                state.error = f"Invalid click index {click_index} for SERP results."
                return state
            click_docid = state.serp.results[click_index-1].docid

            fulltext_or_error = opensearch_client.fetch_fulltext(click_docid)
            if isinstance(fulltext_or_error, Error):
                state.error = f"Failed to fetch full text for {click_docid}: {fulltext_or_error.error_text}"
                return state

            state.fulltext = fulltext_or_error

            instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
            rj_instruction = RelevanceJudgementInstruction(instruction=instruction_text, fulltext=state.fulltext)

            state.relevance_judgement = llm_service.calc_relevance_judgement(model.name, model.temperature, model.top_p, state.memory, rj_instruction)
            qrel_label = qrels.get(state.topic.id, click_docid, default=0)

            output = RelevanceJudgementExperimentOutput(
                session_name = settings.name,
                model = model.name,
                task = settings.task.name,
                dataset = settings.topicset.name,
                topic_id = state.topic.id,
                docid = click_docid,
                label = f"{state.relevance_judgement.label}",
                qrel_label=qrel_label
            )
            print(output.to_json(ensure_ascii=False))

        return state


class QueryReFormulationStage:
    DEFAULT_INSTRUCTION = """
        Review the provided descriptions of task, corpus, tool, search topic, your previous query and search results. Re-Formulate a search query
    """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        if not state.serp or not state.query:
            state.error = "SERP or original query not found, cannot run QueryReFormulationStage."
            return state

        print("\n--- Running: Query Re-formulation Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        qrf_instruction = QueryReFormulationInstruction(instruction=instruction_text)

        state.query = llm_service.recreate_query(model.name, model.temperature, model.top_p, state.memory, qrf_instruction)

        output = QueryReformulationExperimentOutput(
            session_name = settings.name,
            model = model.name,
            task = settings.task.name,
            dataset = settings.topicset.name,
            topic_id = state.topic.id,
            query = state.query.query,
            start = settings.task.start_offset,
            size=settings.task.serp_size
        )
        print(output.to_json(ensure_ascii=False))

        return state

class NextActionStage:
    DEFAULT_INSTRUCTION = """
        Based on the task description, topic details, and your progress so far, decide on the next action to take toward completing the task.

        Choose one of the following actions and provide a brief explanation for your choice:
        - **Submit_New_Query**: Submit a new search query.
        - **Click_Document**: Select a set of documents from the current search results.
        - **GO_NEXT_RESULT_PAGE**: Navigate to the next page of search results for the current query.
        - **End_Task**: End the current task.
    """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        print("\n--- Running: Next Action Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        next_action_instruction = NextActionInstruction(instruction=instruction_text, task=settings.task)

        state.next_action = llm_service.decide_next_action(model.name, model.temperature, model.top_p, state.memory, next_action_instruction)

        action_name = None
        if state.next_action and state.next_action.action:
            action_name = state.next_action.action.name

        output = NextActionOutput(
            session_name = settings.name,
            model = model.name,
            task = settings.task.name,
            dataset = settings.topicset.name,
            topic_id = state.topic.id,
            action = action_name,
            action_num = state.action_num,
            reason = state.next_action.reason
        )
        print(output.to_json(ensure_ascii=False))

        return state

class ExperimentRunner:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings

        self.stage_runners: Dict[str, ExperimentStage] = {
            "query": QueryFormulationStage(self.settings.stages.get("query", StageConfig())),
            "ranking": RankingStage(self.settings.stages.get("ranking", StageConfig())),
            "click": ClickStage(self.settings.stages.get("click", StageConfig())),
            "relevance": RelevanceJudgementStage(self.settings.stages.get("relevance", StageConfig())),
            "reformulate": QueryReFormulationStage(self.settings.stages.get("reformulate", StageConfig())),
            "next_action": NextActionStage(self.settings.stages.get("next_action", StageConfig())),
        }

        self.action_stage_map = {
            Action.SUBMIT_NEW_QUERY: ["query", "ranking", "click", "relevance", "next_action"],
            Action.CLICK_DOCUMENT: ["click", "relevance", "next_action"],
            Action.GO_NEXT_RESULT_PAGE: ["ranking", "click", "relevance", "next_action"],
        }

        self.topic_list_map: Dict[Type[BaseTopic], Type[TopicList]] = {
            TitleOnlyTopic: TopicList[TitleOnlyTopic],
            TitleDescriptionTopic: TopicList[TitleDescriptionTopic],
            TitleNarrativeTopic: TopicList[TitleNarrativeTopic],
            TitleDescriptionNarrativeTopic: TopicList[TitleDescriptionNarrativeTopic],
            FullTopic: TopicList[FullTopic],  # Optional, same as above
        }

        self.llm_factory = LLMServiceFactory()
        self.opensearch_client_factory = OpenSearchClientFactory()
        self._topic_slice: slice | None = self._resolve_topic_slice()
        self.topics = self._load_topics()

    def _resolve_topic_slice(self) -> slice | None:
            """
            Determine which range of topics to load based on ExperimentSettings.

            Priority:
            1) settings.topic_ids: string like '3:10', ':5', '10:'
            2) settings.max_topics: integer (legacy)
            3) None -> no slicing (load all)
            Uses 0-based, end-exclusive Python semantics.
            """
            topic_ids = getattr(self.settings, "topic_ids", None)
            if topic_ids:
                if not isinstance(topic_ids, str) or not re.match(r"^\s*\d*\s*:\s*\d*\s*$", topic_ids):
                    raise ValueError(
                        f"topic_ids must be a slice-like string 'start:end' (digits optional), got {topic_ids!r}"
                    )
                start_str, end_str = topic_ids.split(":")
                start = int(start_str) - 1 if start_str.strip() else None
                stop = int(end_str) if end_str.strip() else None
                return slice(start, stop)

            max_topics = getattr(self.settings, "max_topics", None)
            if isinstance(max_topics, int):
                return slice(0, max_topics)

            return None

    def _load_topics(self) -> TopicList:
        dataset = ir_datasets.load(self.settings.topicset.name)
        if callable(dataset):
            dataset = dataset()

        topic_cls: Type[BaseTopic] = self.settings.topicset.topic_class
        list_cls = self.topic_list_map[topic_cls]

        topics = list_cls()

        query_iter = dataset.queries_iter()
        if self._topic_slice is not None:
            query_iter = islice(query_iter, self._topic_slice.start, self._topic_slice.stop)

        for raw in query_iter:
            topic = topic_cls.from_ir_datasets(raw)
            topics.append(topic)

        return topics
    
    def _get_topic_mapper(self):
        topic_cls: Type[BaseTopic] = self.settings.topicset.topic_class
        name = self.settings.topicset.name.lower()

        if "text" in dir(next(ir_datasets.load(name).queries_iter())):
            # Format: query_id, text
            return lambda q: topic_cls(id=q.query_id, title=q.text)

        # Format: query_id, title, description, narrative
        return lambda q: topic_cls(
            id=q.query_id,
            title=q.title,
            description=getattr(q, "description", None),
            narrative=getattr(q, "narrative", None)
        )

    def run(self):
        print(f"\n{'='*20} Experimental Setting: {self.settings.name} {'='*20}", file=sys.stderr)
        for model in self.settings.models:
            print(f"\n{'='*20} Model: {model.name} ({model.type}) {'='*20}", file=sys.stderr)

            for tool in self.settings.tools:
                print(f"\n{'='*20} Ranker: {tool.ranking_model} ({tool.name}) {'='*20}", file=sys.stderr)
                opensearch_client = self.opensearch_client_factory.create_opensearch_client(settings=self.settings, tool=tool)

                for topic in self.topics:
                    llm_service = self.llm_factory.create_llm_service(model.type)
                    print(f"\n{'--'*10} Topic: {topic.id} ({topic.title}) {'--'*10}", file=sys.stderr)

                    memory = ConversationHistory(system_role=model.system_role, system_prompt=model.system_prompt)
                    state = ExperimentState(topic=topic, memory=memory)

                    state.next_action = NextAction(action=Action.SUBMIT_NEW_QUERY, reason="initial bootstrap")

                    while state.action_num < self.settings.max_actions:
                        stage_names = []

                        if state.next_action and state.next_action.action != Action.END_TASK:
                            action_enum = getattr(state.next_action, "action", None)
                            stage_names = self.action_stage_map.get(action_enum, [])

                        for stage_name in stage_names:
                            if stage_name not in self.stage_runners:
                                print(f"[ERROR] Unknown stage: {stage_name}", file=sys.stderr)
                                sys.exit(1)

                            if stage_name == "ranking" and action_enum == Action.GO_NEXT_RESULT_PAGE:
                                state.query.start += 10

                            stage_runner = self.stage_runners[stage_name]
                            state = stage_runner.run(self.settings, state, llm_service, model, tool, opensearch_client, stage_name)
                            
                            if state.error:
                                print(f"[WARNING] in stage '{stage_name}': {state.error}. Stopping pipeline for this topic.", file=sys.stderr)
                                
                                if self.settings.full_log:
                                    print(f"\n{'--'*10} Full Log {'--'*10}", file=sys.stderr)
                                    all_messages = state.memory.get_all_messages()
                                    pprint.pprint(all_messages, stream=sys.stderr)

                                state.error = None
                                return
                            
                            if stage_name == "next_action" and state.next_action.action == Action.END_TASK:
                                print(f"\n{'='*20} Agent decided to end the task. {'='*20}", file=sys.stderr)

                                if self.settings.full_log:
                                    print(f"\n{'--'*10} Full Log {'--'*10}", file=sys.stderr)
                                    all_messages = state.memory.get_all_messages()
                                    pprint.pprint(all_messages, stream=sys.stderr)
                                
                                return

                            state.action_num += 1

                    if self.settings.full_log:
                        print(f"\n{'--'*10} Full Log {'--'*10}", file=sys.stderr)
                        all_messages = state.memory.get_all_messages()
                        pprint.pprint(all_messages, stream=sys.stderr)