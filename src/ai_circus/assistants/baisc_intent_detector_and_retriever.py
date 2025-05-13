from __future__ import annotations

import json
from typing import Any, Literal

from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from ai_circus.core import custom_logger
from ai_circus.models import get_llm

logger = custom_logger.init(level="INFO")


# State definition
class GraphState(BaseModel):
    """State model for the intent detection graph."""

    user_input: str
    history: list[dict[str, str]] = []
    intent_output: dict[str, Any] = {}


# Intent Detector Prompt Template
INTENT_PROMPT = ChatPromptTemplate.from_template(
    """
**Goal**: Accurately classify the user's intent based on their input and conversation history, reformulate the question for retrieval (if applicable), and detect topic changes.

**Context**: You are an intent detection system that classifies user intents as either "retrieve" (user seeks specific information requiring data retrieval, e.g., facts, definitions, or answers to questions) or "no_retrieve" (user is conversing without needing retrieval, e.g., greetings, opinions, or casual remarks). If the intent is "retrieve," reformulate the question concisely to optimize for a retriever system. Detect if the user has changed the topic significantly from the previous conversation.

**History**:
{conversation_history}

**User Input**: {user_input}

**Examples**:
1. **Input**: "What's the weather in New York?"
   **Output**: ```json
   {{
     "intent": "retrieve",
     "reformulated_question": "Current weather conditions in New York",
     "topic_changed": true
   }}
   ```

2. **Input**: "Hi, how are you?"
   **Output**: ```json
   {{
     "intent": "no_retrieve",
     "reformulated_question": "",
     "topic_changed": false
   }}
   ```

3. **Input**: "Tell me about Python programming" (after asking about weather)
   **Output**: ```json
   {{
     "intent": "retrieve",
     "reformulated_question": "Overview of Python programming language",
     "topic_changed": true
   }}
   ```

4. **Input**: "Can you tell me about the origins of the Python programming language?"
   **Output**: ```json
   {{
     "intent": "retrieve",
     "reformulated_question": "Origins of the Python programming language",
     "topic_changed": true
   }}
   ```

5. **Input**: "What were the key milestones in Python's development?" (after asking about Python's origins)
   **Output**: ```json
   {{
     "intent": "retrieve",
     "reformulated_question": "Key milestones in Python development",
     "topic_changed": false
   }}
   ```

6. **Input**: "What's the weather like today?" (after asking about Python milestones)
   **Output**: ```json
   {{
     "intent": "retrieve",
     "reformulated_question": "Current weather conditions today",
     "topic_changed": true
   }}
   ```

**Instructions**:
- Classify the intent as "retrieve" if the input is a question or a request for specific information (e.g., "Tell me about...", "What is...", "Can you explain..."). Classify as "no_retrieve" only for non-informational inputs (e.g., greetings, opinions, or casual statements like "I love coding").
- For "retrieve" intents, reformulate the question into a concise, keyword-focused query suitable for a retriever, removing conversational fluff (e.g., "Can you tell me about X?" becomes "X").
- For "no_retrieve" intents, set reformulated_question to an empty string ("").
- Set topic_changed to true if the topic differs significantly from the last user input in history (or if history is empty). Set to false if the topic is the same or closely related (e.g., two questions about Python).
- Return a JSON object with exactly these keys: intent, reformulated_question, topic_changed.
- Ensure the response is valid JSON, enclosed in ```json ... ```, with no additional text outside the JSON.

**Output**:
```json
{{
  "intent": "<retrieve or no_retrieve>",
  "reformulated_question": "<reformulated question or empty string>",
  "topic_changed": <true or false>
}}
```
"""
)


# Intent Detector Node
def intent_detector_node(state: GraphState) -> GraphState:
    """Detect intent from user input and update state with the result."""
    llm = get_llm()
    prompt = INTENT_PROMPT.format(conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input)
    # logger.info(f"Formatted prompt:\n{prompt}")
    response = llm.invoke(prompt)
    # logger.info(f"LLM response:\n{response.content}")
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
    required_keys = {"intent", "reformulated_question", "topic_changed"}
    if not all(key in intent_output for key in required_keys):
        logger.error(f"Missing required keys in intent_output: {intent_output}")
        raise ValueError(f"Intent output missing required keys: {required_keys - intent_output.keys()}")
    if intent_output["intent"] not in ["retrieve", "no_retrieve"]:
        logger.error(f"Invalid intent value: {intent_output['intent']}")
        raise ValueError(f"Invalid intent value: {intent_output['intent']}")

    logger.info(f"Parsed intent_output: {intent_output}")
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
    input_history1 = state1.history  # Capture input history
    result1 = graph.invoke(state1)
    state1 = GraphState(**result1)
    logger.info(
        f"Round 1 response:\n{
            json.dumps(
                {'input_history': input_history1, 'question': state1.user_input, 'intent_result': state1.intent_output},
                indent=2,
            )
        }"
    )

    # Round 2: Continue on the same topic with a related question.
    state2 = GraphState(user_input="What were the key milestones in Python's development?", history=state1.history)
    input_history2 = state2.history  # Capture input history
    result2 = graph.invoke(state2)
    state2 = GraphState(**result2)
    logger.info(
        f"Round 2 response:\n{
            json.dumps(
                {'input_history': input_history2, 'question': state2.user_input, 'intent_result': state2.intent_output},
                indent=2,
            )
        }"
    )

    # Round 3: Change the topic with a new question.
    state3 = GraphState(user_input="What's the weather like today?", history=state2.history)
    input_history3 = state3.history  # Capture input history
    result3 = graph.invoke(state3)
    state3 = GraphState(**result3)
    logger.info(
        f"Round 3 response:\n{
            json.dumps(
                {'input_history': input_history3, 'question': state3.user_input, 'intent_result': state3.intent_output},
                indent=2,
            )
        }"
    )
