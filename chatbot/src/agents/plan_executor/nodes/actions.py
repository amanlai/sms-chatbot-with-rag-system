from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .base import BaseNodeChains
from ...prompt_templates import (
    BASE_PROMPT_TEMPLATE,
    TASK_FORMAT_FOR_HUMAN_PROMPT
)
from ...typing import (
    Act,
    LLMOutput,
    Plan,
    PlanExecute as State,
    Response,
)
from utils.config import DEBUG, LOCAL_DEBUG, USE_LLAMA_INDEX


class NodeActions(BaseNodeChains):

    # if DEBUG:
    #     from itertools import count
    #     planner_call_counter_generator = count(0)

    def __init__(self, **kwargs: Any) -> None:
        # if DEBUG:
        #     self.counter = next(self.planner_call_counter_generator)
        super().__init__(**kwargs)

    async def _agent_entry_(
        self, state: State
    ) -> dict[str, list[BaseMessage]]:
        """
        The action taken in the agent node.
        Passes the entire state as input to the replanner chain.

        If the LLM response is Response, then updates the response
        variable of the state. Otherwise, updates the state with
        detailed action steps to take to fully answer the question.
        """
        # this should perhaps be state.get("plan", [])
        # so that the proceeding conditional check is len(plans)>0
        # to safeguard against no plans being produced.
        plans = state.get("plan")
        if plans:
            task = plans[0]
            plans_str = "\n".join(
                [f"{i}. {step}" for i, step in enumerate(plans, 1)]
            )
            task_formatted = TASK_FORMAT_FOR_HUMAN_PROMPT.format(
                input=state["input"],
                plan=plans_str,
                step=1,
                task=task
            )
            messages = [HumanMessage(task_formatted)]
            if LOCAL_DEBUG:
                print(f"----AGENT EXECUTION-----\nTask: {repr(task)}\n")
        else:
            # This is useful if we want to use agent as is.
            # If we're going planner -> agent entry,
            # then we never reach this branch.
            messages = [
                SystemMessage(BASE_PROMPT_TEMPLATE),
                HumanMessage(state["input"])
            ]
        return {"messages": messages}

    def _agent_exit_(
        self, state: State
    ) -> dict[str, list[tuple[str, str]]]:
        last_message = state["messages"][-1]
        last_message_str = last_message.content
        if (plans := state.get("plan")):
            return {"past_steps": [(plans[0], last_message_str)]}
        else:
            if DEBUG:
                print("Returned directly from the agent.\n")
            return {"response": last_message_str}

    def _clear_intermediate_steps_from_previous_agent_run_(
        self, state: State, config: RunnableConfig
    ) -> dict[str, None] | None:
        forget = (
            config
            .get("configurable", {})
            .get("agent_forget_short_memory", True)
        )
        if forget:
            return {"messages": None}
        else:
            return None

    def _fulfill_request_(self, state: State) -> dict[str, str]:
        if LOCAL_DEBUG:
            print("Request\n")
        return {"response": "I got you. Please wait."}

    async def _generate_immediate_answer_(
        self, state: State, config: RunnableConfig
    ) -> dict[str, str]:
        """
        use llm to generate an answer right away
        by looking at past chat history
        """
        output: LLMOutput = await self._answerer.ainvoke(
            state,
            config=config,
            stream_mode="values"
        )
        if LOCAL_DEBUG:
            print("----- IMMEDIATE ANSWER -----\n")
            print(f"{repr(output)}\n\n")
        return {"response": output.pop("output"), **output}

    async def _call_o1_(
        self, state: State, config: RunnableConfig
    ) -> dict[str, str]:
        output: LLMOutput = await self._agent_llm.ainvoke(
            state,
            config=config,
            stream_mode="values"
        )
        response = output.pop("output")
        if LOCAL_DEBUG:
            print(f"-------AGENT OUTPUT-------\n{repr(response)}\n")
        return {"response": response, **output}

    async def _planner_(
        self, state: State, config: RunnableConfig
    ) -> dict[str, list[str] | str]:
        """
        The action taken in the planner node.
        Only passes the input (with question and context)
        as input to the planner chain.

        Updates the state with detailed action steps to take
        to fully answer the question.
        """
        output: LLMOutput = await self._planner.ainvoke(
            state,
            config=config,
            stream_mode="values"
        )
        plan = output.pop("output")
        if LOCAL_DEBUG:
            print(f"-------PLAN-------\n{repr(plan)}\n")
        if isinstance(plan, Plan):
            return {"plan": plan.steps, **output}
        elif isinstance(plan, BaseMessage):
            # if the LLM result is not a Plan object, that means it
            # resulted in error.
            # we need to handle this error differently.
            # e.g. we probably need to connect to an actual person to look
            # into what's wrong.
            # for now, we just ignore it and go on
            return {"plan": [plan.content], **output}
        else:
            # in what case can this be a string?
            return {"plan": [plan], **output}

    async def _process_query_(
        self, state: State, config: RunnableConfig
    ) -> dict[str, str]:
        """use llm to make the question better if possible"""
        output: LLMOutput = await self._query_processor.ainvoke(
            state,
            config=config,
            stream_mode="values"
        )
        if LOCAL_DEBUG:
            # print(f"----- PLAN EXECUTOR #{self.counter} -----")
            print(f"-------ENTRY-------\n-----State------\n{repr(state)}\n")
            print(f"-------PROCESSED QUESTION-------\n{repr(output)}\n")
        return {"input": output.pop("output"), "past_steps": None, **output}

    async def _replanner_(
        self, state: State, config: RunnableConfig
    ) -> dict[str, str | list[str]]:
        """
        The action taken in the replan node.
        Passes the entire state as input to the replanner chain.

        If the LLM response is Response, then updates the response
        variable of the state. Otherwise, updates the state with
        detailed action steps to take to fully answer the question.
        """
        output: LLMOutput = await self._replanner.ainvoke(
            state,
            config=config,
            stream_mode="values"
        )
        if LOCAL_DEBUG:
            print(f"-------REPLAN-------\n{repr(output)}\n")
        act = output.pop("output")
        if isinstance(act, Act):
            if isinstance(act.action, Response):
                return {"response": act.action.response, **output}
            else:
                return {"plan": act.action.steps, **output}
        elif isinstance(act, BaseMessage):
            # if the LLM result is not an Act object, that means it
            # resulted in error.
            # we need to handle this error differently.
            # e.g. we probably need to connect to an actual person to look
            # into what's wrong.
            # for now, we just ignore it and respond to the user
            return {"response": act.content, **output}
        else:
            # if it's a string. in what case can this happen?
            return {"response": act, **output}

    async def _retriever_(self, state: State) -> dict[str, str | None]:
        documents = await self._retriever.ainvoke(state["input"])
        formatted_docs = await self._format_docs(documents)
        # input = self._format_template(state["input"], formatted_docs)
        if DEBUG:
            print("-------RETRIEVED DOCUMENT SCORES-------")
            if USE_LLAMA_INDEX:
                print(*(f"{x.node.id_}: {x.score}\n" for x in documents))
            else:
                print(*(f'{x.metadata["_id"]}: {x.metadata["score"]}\n' for x in documents))  # noqa: E501
        return {"retrieved_context": formatted_docs, "response": None}
