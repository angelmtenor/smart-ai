"""Basic intent detector and retriever.
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Literal

import yaml
from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from ai_circus.core import custom_logger
from ai_circus.models import get_llm

logger = custom_logger.init(level="INFO")


# Load prompt template from YAML file
def load_prompt_template(node: str) -> str:
    """Load the prompt template for the specified node from ai_circus.assistants.prompts.yaml."""
    try:
        with resources.files("ai_circus.assistants").joinpath("prompts.yaml").open("r") as file:
            config = yaml.safe_load(file)
        node_config = config.get(node, {})
        shared_fields = {
            "scenario_description": config.get("scenario_description", ""),
            "scenario_documents": config.get("scenario_documents", ""),
        }

        # Combine relevant fields into a single prompt template, ensuring scenario context is always included
        prompt_parts = [
            node_config.get("goal", ""),
            node_config.get("output_format", ""),
            node_config.get("instructions", ""),
            shared_fields["scenario_description"],
            shared_fields["scenario_documents"],
            node_config.get("history", ""),
            node_config.get("user_input", ""),
            node_config.get("examples", ""),
        ]
        prompt = "\n\n".join(part for part in prompt_parts if part)

        # Escape curly braces, preserving intended placeholders
        placeholders = ["{conversation_history}", "{user_input}"]
        for placeholder in placeholders:
            prompt = prompt.replace(placeholder, f"__TEMP_{placeholder[1:-1]}__")
        prompt = prompt.replace("{", "{{").replace("}", "}}")
        for placeholder in placeholders:
            temp_key = f"__TEMP_{placeholder[1:-1]}__"
            prompt = prompt.replace(temp_key, placeholder)

        return prompt
    except FileNotFoundError:
        logger.error("Prompt YAML file not found in ai_circus.assistants")
        raise FileNotFoundError("Could not find prompts.yaml in ai_circus.assistants") from None
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file: {e}")
        raise ValueError(f"Invalid YAML format in prompts.yaml: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load prompts.yaml from package: {e}")
        raise ValueError(f"Error accessing prompts.yaml: {e}") from e


# Prompt Templates
INTENT_PROMPT = ChatPromptTemplate.from_template(load_prompt_template("intent_detection"))
NON_RETRIEVER_PROMPT = ChatPromptTemplate.from_template(load_prompt_template("non_retriever_response"))
POST_RETRIEVER_PROMPT = ChatPromptTemplate.from_template(load_prompt_template("post_retriever_response"))


# State definition
class GraphState(BaseModel):
    """State model for the intent detection and response graph."""

    user_input: str
    history: list[dict[str, str]] = []
    intent_output: dict[str, Any] = {}
    response_output: dict[str, Any] = {}


# Intent Detector Node
def intent_detector_node(state: GraphState) -> GraphState:
    """Detect intent from user input and update state with the result."""
    llm = get_llm()
    prompt = INTENT_PROMPT.format(conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input)
    response = llm.invoke(prompt)
    content = str(response.content).strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    try:
        intent_output = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {content}")
        raise ValueError(f"Invalid JSON response from LLM: {e}") from e

    # Validate intent_output
    if not isinstance(intent_output, dict):
        logger.error(f"Expected intent_output to be a dict, got: {intent_output}")
        raise ValueError("Expected response content to be a dict")
    required_keys = {"intent", "reformulated_question", "new_topic"}
    if not all(key in intent_output for key in required_keys):
        logger.error(f"Missing required keys in intent_output: {intent_output}")
        raise ValueError(f"Intent output missing required keys: {required_keys - intent_output.keys()}")
    if intent_output["intent"] not in ["retrieve", "no_retrieve"]:
        logger.error(f"Invalid intent value: {intent_output['intent']}")
        raise ValueError(f"Invalid intent value: {intent_output['intent']}")

    state.intent_output = intent_output
    return state


# Non-Retriever Response Node
def non_retriever_response_node(state: GraphState) -> GraphState:
    """Generate a conversational response for non-retrieval intents."""
    llm = get_llm()
    prompt = NON_RETRIEVER_PROMPT.format(
        conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input
    )
    response = llm.invoke(prompt)
    content = str(response.content).strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    try:
        response_output = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {content}")
        raise ValueError(f"Invalid JSON response from LLM: {e}") from e

    # Validate response_output
    if not isinstance(response_output, dict):
        logger.error(f"Expected response_output to be a dict, got: {response_output}")
        raise ValueError("Expected response content to be a dict")
    required_keys = {"response", "is_within_scope"}
    if not all(key in response_output for key in required_keys):
        logger.error(f"Missing required keys in response_output: {response_output}")
        raise ValueError(f"Response output missing required keys: {required_keys - response_output.keys()}")

    state.response_output = response_output
    state.history.append({"user": state.user_input, "assistant": response_output["response"]})
    return state


# Post-Retriever Response Node
def post_retriever_response_node(state: GraphState) -> GraphState:
    """Generate a detailed response for retrieval intents based on reformulated question."""
    llm = get_llm()
    prompt = POST_RETRIEVER_PROMPT.format(
        conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input
    )
    response = llm.invoke(prompt)
    content = str(response.content).strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    try:
        response_output = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {content}")
        raise ValueError(f"Invalid JSON response from LLM: {e}") from e

    # Validate response_output
    if not isinstance(response_output, dict):
        logger.error(f"Expected response_output to be a dict, got: {response_output}")
        raise ValueError("Expected response content to be a dict")
    required_keys = {"response", "is_within_scope"}
    if not all(key in response_output for key in required_keys):
        logger.error(f"Missing required keys in response_output: {response_output}")
        raise ValueError(f"Response output missing required keys: {required_keys - response_output.keys()}")

    state.response_output = response_output
    state.history.append({"user": state.user_input, "assistant": response_output["response"]})
    return state


# Conditional Edge Logic
def route_intent(state: GraphState) -> Literal["post_retriever", "non_retriever"]:
    """Route to the appropriate node based on detected intent."""
    intent = state.intent_output.get("intent", "no_retrieve")
    return "post_retriever" if intent == "retrieve" else "non_retriever"


# Build the Graph
def build_graph() -> Runnable:
    """Compile and return the intent detection and response workflow graph."""
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


# Example Usage
if __name__ == "__main__":
    graph = build_graph()

    # Round 1: Ask about Python programming
    state1 = GraphState(user_input="Tell me about Python programming", history=[])
    input_history1 = state1.history
    result1 = graph.invoke(state1)
    state1 = GraphState(**result1)
    logger.info(
        f"Round 1 response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history1,
                    'question': state1.user_input,
                    'intent_result': state1.intent_output,
                    'response': state1.response_output,
                },
                indent=2,
            )
        }"
    )

    # Round 2: Ask an incomplete question to test reformulated question
    state2 = GraphState(user_input="Who created it?", history=state1.history)
    input_history2 = state2.history
    result2 = graph.invoke(state2)
    state2 = GraphState(**result2)
    logger.info(
        f"Round 2 response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history2,
                    'question': state2.user_input,
                    'intent_result': state2.intent_output,
                    'response': state2.response_output,
                },
                indent=2,
            )
        }"
    )

    # Round 3: Continue on the same topic with a related question
    state3 = GraphState(user_input="What are its key milestones?", history=state2.history)
    input_history3 = state3.history
    result3 = graph.invoke(state3)
    state3 = GraphState(**result3)
    logger.info(
        f"Round 3 response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history3,
                    'question': state3.user_input,
                    'intent_result': state3.intent_output,
                    'response': state3.response_output,
                },
                indent=2,
            )
        }"
    )

    # Round 4: Non-retrieval input
    state4 = GraphState(user_input="I love coding!", history=state3.history)
    input_history4 = state4.history
    result4 = graph.invoke(state4)
    state4 = GraphState(**result4)
    logger.info(
        f"Round 4 response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history4,
                    'question': state4.user_input,
                    'intent_result': state4.intent_output,
                    'response': state4.response_output,
                },
                indent=2,
            )
        }"
    )

    # Round 5: Ask about a different topic (weather)
    state5 = GraphState(user_input="What is the weather like today?", history=state4.history)
    input_history5 = state5.history
    result5 = graph.invoke(state5)
    state5 = GraphState(**result5)
    logger.info(
        f"Round 5 response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history5,
                    'question': state5.user_input,
                    'intent_result': state5.intent_output,
                    'response': state5.response_output,
                },
                indent=2,
            )
        }"
    )

    # Round 6: Ask about what the assistant can do
    state6 = GraphState(user_input="What can you do?", history=state5.history)
    input_history6 = state6.history
    result6 = graph.invoke(state6)
    state6 = GraphState(**result6)
    logger.info(
        f"Round 6 response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history6,
                    'question': state6.user_input,
                    'intent_result': state6.intent_output,
                    'response': state6.response_output,
                },
                indent=2,
            )
        }"
    )

    # round 7 summarize the whole conversation
    state7 = GraphState(user_input="Summarize our conversation", history=state6.history)
    input_history7 = state7.history
    result7 = graph.invoke(state7)
    state7 = GraphState(**result7)
    logger.info(
        f"Round 7 response:\n"
        f"{
            json.dumps(
                {
                    'input_history': input_history7,
                    'question': state7.user_input,
                    'intent_result': state7.intent_output,
                    'response': state7.response_output,
                },
                indent=2,
            )
        }"
    )
