from datetime import datetime
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage

from langgraph.graph import add_messages
from langgraph.graph.message import Messages
from langgraph.managed import IsLastStep

from pydantic import BaseModel, Field


Pairs = list[tuple[str, str]]


def concatenate_past_steps(left: Pairs, right: Pairs | None) -> Pairs:
    if isinstance(right, list):
        return left + right
    else:
        return []


def manage_messages(left: Messages, right: Messages | None) -> Messages:
    if isinstance(right, list):
        return add_messages(left, right)
    else:
        return []


class ConfigSchema(TypedDict):
    thread_id: str
    forget: bool = False
    trim_intermediate_steps: bool = True


class BaseState(TypedDict):
    """
    The state of the agent
    """
    input: str
    response: str
    chat_history: list[BaseMessage]
    retrieved_context: str
    error: str


class LLMOutput(TypedDict):
    """
    Output of ChatModelWithErrorHandling
    """
    output: BaseMessage | BaseModel | str
    error: str


class PlanExecute(BaseState):
    """
    The state of the plan-execute agent
    """
    messages: Annotated[list[BaseMessage], manage_messages]
    is_last_step: IsLastStep
    input: str
    plan: list[str]
    past_steps: Annotated[Pairs, concatenate_past_steps]
    response: str
    chat_history: list[BaseMessage]
    retrieved_context: str
    error: str


class ClassifyQuestion(BaseModel):
    """
    Binary score to assess if the question is about hotel amenities,
    services or surrounding areas.
    """
    binary_score: Literal["yes", "no"] = Field(
        description="Question is related to hotel, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """
    Binary score to assess if answer addresses question.
    """
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class Plan(BaseModel):
    """
    Plan to follow in future
    """
    steps: list[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """
    Response to user.
    """
    response: str


class Act(BaseModel):
    """
    Action to perform.
    """
    action: Response | Plan = Field(
        description="Action to perform. "
        "If you want to respond to user, use Response. "
        "Otherwise, use Plan."
    )


class RetrieverInput(BaseModel):
    """
    Input to the retriever.
    """
    query: str = Field(
        description="query to look up in retriever"
    )


class RouteQuestion(BaseModel):
    """
    Binary classifier to check if an input question is a request or not.
    """
    binary_score: Literal["yes", "no"] = Field(
        description="The question is a request for an item, 'yes' or 'no'"
    )


class TwilioResponseMessage(BaseModel):
    """
    Message reply from Twilio
    """
    status: str
    sid: str | None
    date_created: datetime
    body: str | None
