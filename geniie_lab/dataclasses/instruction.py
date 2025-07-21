# Standard library
from dataclasses import dataclass
from textwrap import dedent
from typing import Union

# Local application imports
from geniie_lab.dataclasses.description import (
    CorpusDescription,
    TaskDescription,
    ToolDescription,
)
from geniie_lab.dataclasses.serp import FullText, Serp
from geniie_lab.dataclasses.topic import (
    TitleDescriptionNarrativeTopic, FullTopic,
    TitleDescriptionTopic,
    TitleNarrativeTopic,
    TitleOnlyTopic
)

@dataclass
class QueryFormulationInstruction:
    instruction: str
    task: TaskDescription
    corpus: CorpusDescription
    tool: ToolDescription
    topic: Union[TitleOnlyTopic, TitleDescriptionTopic, TitleNarrativeTopic, TitleDescriptionNarrativeTopic, FullTopic]

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Task Description**: {self.task.description}
            **Corpus Description**: {self.corpus.description}
            **Search Tool Description**: {self.tool.description}
            **Topic Description**: {self.topic}
        """
        return dedent(instruction).strip()

@dataclass
class ClickInstruction:
    instruction: str
    serp: Serp

    def generate(self) -> str:
        content = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Search results**:
            {self.serp.results}

            **Note**: Before response, ensure that all numbers in ranking_list match in the search results.
        """
        return dedent(content).strip()

@dataclass
class RelevanceJudgementInstruction:
    instruction: str
    fulltext: FullText

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Document**: {self.fulltext.title}
            {self.fulltext.text}
        """
        return dedent(instruction).strip()

@dataclass
class QueryReFormulationInstruction:
    instruction: str

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
        """
        return dedent(instruction).strip()


@dataclass
class NextActionInstruction:
    instruction: str
    task: TaskDescription

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Task Description**: {self.task.description}
        """
        return dedent(instruction).strip()