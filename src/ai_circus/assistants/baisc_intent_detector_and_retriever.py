from __future__ import annotations

import json
from typing import Any, Literal

import yaml
from importlib import resources
from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from ai_circus.core import custom_logger
from ai_circus.models import get_llm

logger = custom_logger.init(level="INFO")


# Load prompt template from YAML file
def load_prompt_template() -> str:
    """Load the intent detection prompt template from ai_circus.assistants.prompts.yaml."""
    try:
        # Access prompts.yaml as a package resource
        with resources.files("ai_circus.assistants").joinpath("prompts.yaml").open("r") as file:
            config = yaml.safe_load(file)
        intent_config = config.get("intent_detection", {})

        # Combine relevant fields into a single prompt template
        prompt_parts = [
            intent_config.get("goal", ""),
            intent_config.get("context_scenario", ""),
            intent_config.get("context_documents", ""),
            intent_config.get("history", ""),
            intent_config.get("user_input", ""),
            intent_config.get("examples", ""),
            intent_config.get("instructions", ""),
            intent_config.get("output_format", "")
        ]
        prompt = "\n\n".join(part for part in prompt_parts if part)

        # Escape curly braces, preserving intended placeholders
        # Replace { with {{ and } with }}, except for {conversation_history} and {user_input}
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
        raise FileNotFoundError("Could not find prompts.yaml in ai_circus.assistants")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file: {e}")
        raise ValueError(f"Invalid YAML format in prompts.yaml: {e}")
    except Exception as e:
        logger.error(f"Failed to load prompts.yaml from package: {e}")
        raise ValueError(f"Error accessing prompts.yaml: {e}")


# Intent Detector Prompt Template
INTENT_PROMPT = ChatPromptTemplate.from_template(load_prompt_template())


# State definition
class GraphState(BaseModel):
    """State model for the intent detection graph."""
    user_input: str
    history: list[dict[str, str]] = []
    intent_output: dict[str, Any] = {}


# Intent Detector Node
def intent_detector_node(state: GraphState) -> GraphState:
    """Detect intent from user input and update state with the result."""
    llm = get_llm()
    prompt = INTENT_PROMPT.format(conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input)
    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    try:
        intent_output = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {content}")
        raise ValueError(f"Invalid JSON response from LLM: {e}")

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

    # logger.info(f"Parsed intent_output: {intent_output}")
    state.intent_output = intent_output
    return state


# Retriever Node
def retriever_node(state: GraphState) -> GraphState:
    """Handle retrieval intent and simulate retrieval logic."""
    reformulated_question = state.intent_output.get("reformulated_question", "")
    retrieval_result = f"Retrieved data for: {reformulated_question}"
    state.history.append({"user": state.user_input, "assistant": retrieval_result})
    return state


# Non-Retriever Node
def non_retriever_node(state: GraphState) -> GraphState:
    """Handle non-retrieval intent with a conversational response."""
    response = "I understand, let's continue the conversation!"
    state.history.append({"user": state.user_input, "assistant": response})
    return state


# Conditional Edge Logic
def route_intent(state: GraphState) -> Literal["retriever", "non_retriever"]:
    """Route to the appropriate node based on detected intent."""
    intent = state.intent_output.get("intent", "no_retrieve")
    return "retriever" if intent == "retrieve" else "non_retriever"


# Build the Graph
def build_graph() -> Runnable:
    """Compile and return the intent detection workflow graph."""
    workflow = StateGraph(GraphState)
    workflow.add_node("intent_detector", intent_detector_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("non_retriever", non_retriever_node)
    workflow.set_entry_point("intent_detector")
    workflow.add_conditional_edges(
        "intent_detector", route_intent, {"retriever": "retriever", "non_retriever": "non_retriever"}
    )
    workflow.add_edge("retriever", END)
    workflow.add_edge("non_retriever", END)
    return workflow.compile()


# Example Usage
if __name__ == "__main__":
    graph = build_graph()

    # Round 1: Ask about the origins of Python in history.
    state1 = GraphState(user_input="Can you tell me about the origins of the Python programming language?", history=[])
    input_history1 = state1.history
    result1 = graph.invoke(state1)
    state1 = GraphState(**result1)
    logger.info(
        f"Round 1 response:\n{json.dumps({'input_history': input_history1, 'question': state1.user_input, 'intent_result': state1.intent_output}, indent=2)}"
    )

    # Round 2: Continue on the same topic with a related question.
    state2 = GraphState(user_input="What were the key milestones there?", history=state1.history)
    input_history2 = state2.history
    result2 = graph.invoke(state2)
    state2 = GraphState(**result2)
    logger.info(
        f"Round 2 response:\n{json.dumps({'input_history': input_history2, 'question': state2.user_input, 'intent_result': state2.intent_output}, indent=2)}"
    )

    # Round 3: Change the topic with a new question.
    state3 = GraphState(user_input="What's the weather like today?", history=state2.history)
    input_history3 = state3.history
    result3 = graph.invoke(state3)
    state3 = GraphState(**result3)
    logger.info(
        f"Round 3 response:\n{json.dumps({'input_history': input_history3, 'question': state3.user_input, 'intent_result': state3.intent_output}, indent=2)}"
    )
