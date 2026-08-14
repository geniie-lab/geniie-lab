from enum import Enum
from typing import List

from pydantic import BaseModel, Field

# Enums
class Relevance(str, Enum):
    """Document relevance enumeration."""
    RELEVANT = "Relevant"
    NOT_RELEVANT = "NotRelevant"

class RubricRelevance(str, Enum):
    """What happened to the information need a subtopic expresses."""
    NOT_ADDRESSED = "NotAddressed"
    ON_SUBTOPIC_ONLY = "OnSubtopicOnly"
    NEED_SATISFIED = "NeedSatisfied"
    COMPLETELY_SATISFIED = "CompletelySatisfied"

# Rubric order, ascending. Use for "at least NEED_SATISFIED" comparisons.
RUBRIC_ORDER = (
    RubricRelevance.NOT_ADDRESSED,
    RubricRelevance.ON_SUBTOPIC_ONLY,
    RubricRelevance.NEED_SATISFIED,
    RubricRelevance.COMPLETELY_SATISFIED,
)

class Action(Enum):
    """Enum describing possible user actions."""
    SUBMIT_NEW_QUERY = "SUBMIT_NEW_QUERY"
    CLICK_DOCUMENT = "CLICK_DOCUMENT"
    GO_NEXT_RESULT_PAGE = "GO_NEXT_RESULT_PAGE"
    END_TASK = "END_TASK"

# Models
class Query(BaseModel):
    """A model for submitting a query to a search tool."""
    query: str = Field(
        ...,
        title="query",
        description="The query string submitted to the search tool."
    )
    start: int = Field(
        0,
        title="start",
        description=(
            "The starting index of the search results. Defaults to 0. "
        )
    )
    size: int = Field(
        10,
        title="size",
        description=(
            "The number of documents per search result page. Defaults to 10. "
        )
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation of the intent behind your query."
    )

class Clicks(BaseModel):
    """
    A model for selecting multiple documents from search results.
    """
    ranking_list: List[int] = Field(
        ...,
        title="ranking_list",
        description=(
            "The ranking numbers of the documents in the result to examine the full text. "
            "Each element is one ranking number, e.g. [2, 5, 7]. "
            "Use ranking numbers, not document IDs."
        ),
        # Plain integer items. A string coercion with a digit pattern used to
        # live here as a Groq workaround (its decoder dropped commas between
        # bare integers). Groq is no longer used, and the string grammar broke
        # gpt-oss on vLLM: guided decoding blocked the model's natural `[1`
        # continuation and it degenerated to an empty (but valid) list, i.e.
        # silent zero-click sessions. Integer typing also keeps document-ID
        # strings impossible under grammar-enforcing providers.
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation for selecting these documents."
    )

class RelevanceJudgement(BaseModel):
    """A model for labeling a document's relevance."""
    label: Relevance = Field(
        ...,
        title="label",
        description=(
            "The relevance label of the document based on the information need "
            "specified in the topic file."
        )
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation supporting your judgment."
    )

class NextAction(BaseModel):
    """A model for specifying the next step toward completing a task."""
    action: Action = Field(
        ...,
        title="action",
        description="The next step to take toward completing the given task."
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation for choosing this action."
    )

class SubtopicRelevance(BaseModel):
    """One subtopic's relevance label for a single document."""
    subtopic: int = Field(
        ...,
        title="subtopic",
        description="The number of the subtopic as listed in the search topic."
    )
    label: RubricRelevance = Field(
        ...,
        title="label",
        description="The relevance label for this subtopic, as defined in the instruction."
    )
    evidence: str = Field(
        "",
        title="evidence",
        description=(
            "A verbatim quotation copied from the document that supports this "
            "label: the quoted text only, with no commentary or explanation "
            "(put those in the reason field). Empty string when the label is "
            "NotAddressed."
        )
    )

class SubtopicRelevanceJudgement(BaseModel):
    """A model for labelling a document against every subtopic of the search
    topic, one entry per listed subtopic."""
    labels: List[SubtopicRelevance] = Field(
        ...,
        title="labels",
        description=(
            "One entry per subtopic listed in the search topic, in the listed "
            "order, each with its label and supporting quotation."
        )
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation supporting your judgment."
    )
