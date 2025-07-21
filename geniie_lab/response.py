from enum import Enum
from typing import List

from pydantic import BaseModel, Field

# Enums
class Relevance(str, Enum):
    """Document relevance enumeration."""
    RELEVANT = "Relevant"
    NOT_RELEVANT = "NotRelevant"

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
            "The ranking number of the documents in the result to examine the full text. "
        )
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
