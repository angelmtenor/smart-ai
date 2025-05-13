"""
Basic intent detector and retriever.
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template

Note: This script expects a `prompts.yaml` file in `ai_circus.assistants`
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Literal

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from ai_circus.core import custom_logger
from ai_circus.models import get_llm

logger = custom_logger.init(level="DEBUG")


def load_prompt_template(node: str) -> str:
    """Load prompt template for the specified node from prompts.yaml and add tags with carriage returns."""
    try:
        with resources.files("ai_circus.assistants").joinpath("prompts.yaml").open("r") as file:
            config = yaml.safe_load(file)
        node_config = config.get(node, {})
        if not node_config:
            logger.error(f"No configuration found for node: {node}")
            raise ValueError(f"No configuration found for node: {node}")
        shared_fields = {
            "scenario_description": config.get("scenario_description", ""),
            "scenario_documents": config.get("scenario_documents", ""),
            "scenario_sample_questions": config.get("scenario_sample_questions", ""),
        }

        # Define tags for each section
        tag_map = {
            "goal": ("<GOAL>", "</GOAL>"),
            "output_format": ("<OUTPUT_FORMAT>", "</OUTPUT_FORMAT>"),
            "instructions": ("<INSTRUCTIONS>", "</INSTRUCTIONS>"),
            "history": ("<HISTORY>", "</HISTORY>"),
            "user_input": ("<USER_INPUT>", "</USER_INPUT>"),
            "examples": ("<EXAMPLES>", "</EXAMPLES>"),
            "scenario_description": ("<SCENARIO_DESCRIPTION>", "</SCENARIO_DESCRIPTION>"),
            "scenario_documents": ("<SCENARIO_DOCUMENTS>", "</SCENARIO_DOCUMENTS>"),
            "scenario_sample_questions": ("<SCENARIO_SAMPLE_QUESTIONS>", "</SCENARIO_SAMPLE_QUESTIONS>"),
        }

        # Wrap each section with its tags and carriage returns
        def wrap_with_tags(content: str, key: str) -> str:
            if not content:
                return ""
            start_tag, end_tag = tag_map.get(key, ("", ""))
            return f"{start_tag}\r\n{content.rstrip()}\r\n{end_tag}"

        prompt_parts = [
            wrap_with_tags(node_config.get("goal", ""), "goal"),
            wrap_with_tags(node_config.get("output_format", ""), "output_format"),
            wrap_with_tags(node_config.get("instructions", ""), "instructions"),
            wrap_with_tags(shared_fields["scenario_description"], "scenario_description"),
            wrap_with_tags(shared_fields["scenario_documents"], "scenario_documents"),
            wrap_with_tags(shared_fields["scenario_sample_questions"], "scenario_sample_questions"),
            wrap_with_tags(node_config.get("history", ""), "history"),
            wrap_with_tags(node_config.get("user_input", ""), "user_input"),
            wrap_with_tags(node_config.get("examples", ""), "examples"),
        ]
        prompt = "\r\n\r\n".join(part for part in prompt_parts if part)

        # Escape curly braces, preserving placeholders
        placeholders = ["{conversation_history}", "{user_input}"]
        for placeholder in placeholders:
            prompt = prompt.replace(placeholder, f"__TEMP_{placeholder[1:-1]}__")
        prompt = prompt.replace("{", "{{").replace("}", "}}")
        for placeholder in placeholders:
            temp_key = f"__TEMP_{placeholder[1:-1]}__"
            prompt = prompt.replace(temp_key, placeholder)

        if not prompt.strip():
            logger.error(f"Empty prompt generated for node: {node}")
            raise ValueError(f"Empty prompt for node: {node}")

        logger.debug(f"Loaded prompt for {node}:\n\n {prompt}")
        return prompt
    except FileNotFoundError as e:
        logger.error("Prompt YAML file not found")
        raise FileNotFoundError("Could not find prompts.yaml") from e
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format: {e}")
        raise ValueError(f"Failed to parse prompts.yaml: {e}") from e
    except Exception as e:
        logger.error(f"Error loading prompts.yaml: {e}")
        raise ValueError(f"Failed to load prompts.yaml: {e}") from e


# Prompt Templates
INTENT_PROMPT = ChatPromptTemplate.from_template(load_prompt_template("intent_detection"))
NON_RETRIEVER_PROMPT = ChatPromptTemplate.from_template(load_prompt_template("non_retriever_response"))
POST_RETRIEVER_PROMPT = ChatPromptTemplate.from_template(load_prompt_template("post_retriever_response"))


class GraphState(BaseModel):
    """State model for intent detection and response graph."""

    user_input: str
    history: list[dict[str, str]] = []
    intent_output: dict[str, Any] = {}
    response_output: dict[str, Any] = {}


def process_llm_response(content: str) -> dict:
    """Parse and validate LLM JSON response."""
    content = content.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response: {content}")
        raise ValueError(f"Failed to parse LLM response: {e}") from e


def validate_output(output: dict, required_keys: set[str], node: str) -> None:
    """Validate output dictionary structure."""
    if not isinstance(output, dict):
        logger.error(f"Expected {node} output to be dict, got: {output}")
        raise ValueError(f"Invalid {node} output type")
    missing_keys = required_keys - set(output.keys())
    if missing_keys:
        logger.error(f"Missing keys in {node} output: {missing_keys}")
        raise ValueError(f"Missing required keys: {missing_keys}")


def intent_detector_node(state: GraphState) -> GraphState:
    """Detect intent from user input."""
    llm = get_llm()
    prompt = INTENT_PROMPT.format(conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input)
    response = llm.invoke(prompt)
    intent_output = process_llm_response(str(response.content))
    validate_output(intent_output, {"intent", "reformulated_question", "new_topic"}, "intent_detector")
    if intent_output["intent"] not in ["retrieve", "no_retrieve"]:
        logger.error(f"Invalid intent: {intent_output['intent']}")
        raise ValueError(f"Invalid intent value: {intent_output['intent']}")
    state.intent_output = intent_output
    return state


def non_retriever_response_node(state: GraphState) -> GraphState:
    """Generate response for non-retrieval intents."""
    llm = get_llm()
    prompt = NON_RETRIEVER_PROMPT.format(
        conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input
    )
    response = llm.invoke(prompt)
    response_output = process_llm_response(str(response.content))
    validate_output(response_output, {"response", "is_within_scope"}, "non_retriever")
    state.response_output = response_output
    state.history.append({"user": state.user_input, "assistant": response_output["response"]})
    return state


def post_retriever_response_node(state: GraphState) -> GraphState:
    """Generate response for retrieval intents."""
    llm = get_llm()
    prompt = POST_RETRIEVER_PROMPT.format(
        conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input
    )
    response = llm.invoke(prompt)
    response_output = process_llm_response(str(response.content))
    validate_output(response_output, {"response", "is_within_scope"}, "post_retriever")
    state.response_output = response_output
    state.history.append({"user": state.user_input, "assistant": response_output["response"]})
    return state


def route_intent(state: GraphState) -> Literal["post_retriever", "non_retriever"]:
    """Route based on detected intent."""
    return "post_retriever" if state.intent_output.get("intent") == "retrieve" else "non_retriever"


def build_graph() -> CompiledStateGraph:
    """Compile the intent detection and response workflow graph."""
    workflow = StateGraph(GraphState)
    workflow.add_node("intent_detector", intent_detector_node)
    workflow.add_node("post_retriever", post_retriever_response_node)
    workflow.add_node("non_retriever", non_retriever_response_node)
    workflow.set_entry_point("intent_detector")
    workflow.add_conditional_edges(
        "intent_detector", route_intent, {"post_retriever": "post_retriever", "non_retriever": "non_retriever"}
    )
    workflow.add_edge("post_retriever", END)
    workflow.add_edge("non_retriever", END)
    return workflow.compile()


def log_round(round_num: int, state: GraphState, input_history: list) -> None:
    """Log round details in a structured format."""
    logger.info(
        f"Round {round_num} response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history,
                    'question': state.user_input,
                    'intent_result': state.intent_output,
                    'response': state.response_output,
                },
                indent=2,
            )
        }"
    )


if __name__ == "__main__":
    graph = build_graph()
    history = []

    # Round 1: Retrieval intent (Python-related)
    state = GraphState(user_input="Tell me about Python programming", history=history)
    result = graph.invoke(state)
    state = GraphState(**result)
    history = state.history
    log_round(1, state, history[:-1])

    # Round 2: Related retrieval intent (same topic)
    state = GraphState(user_input="What are its key milestones?", history=history)
    result = graph.invoke(state)
    state = GraphState(**result)
    history = state.history
    log_round(2, state, history[:-1])

    # Round 3: Non-retrieval intent (chit-chat)
    state = GraphState(user_input="I love coding!", history=history)
    result = graph.invoke(state)
    state = GraphState(**result)
    history = state.history
    log_round(3, state, history[:-1])

    # Round 4: Out-of-scope intent (weather)
    state = GraphState(user_input="What is the weather like today?", history=history)
    result = graph.invoke(state)
    state = GraphState(**result)
    history = state.history
    log_round(4, state, history[:-1])
