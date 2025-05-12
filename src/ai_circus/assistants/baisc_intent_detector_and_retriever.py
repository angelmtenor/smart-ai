"""Basic Intent Detector and Retriever -- DRAFT - IN PROGRESS
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

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
**Goal**: Analyze the user's input and conversation history to determine the intent, reformulate the question for
 retrieval (if applicable), and detect topic changes.

**Context**: You are an intent detection system that classifies user intents as either "retrieve"
(user seeks information requiring data retrieval) or "no_retrieve" (user is conversing without needing retrieval,
 e.g., greetings, opinions). If the intent is "retrieve," reformulate the question to optimize for a retriever system.
 Detect if the user has changed the topic from the previous conversation.

**History**:
{conversation_history}

**User Input**: {user_input}

**Examples**:
1. **Input**: "What's the weather in New York?"
   **Output**: ```json
   {
     "intent": "retrieve",
     "reformulated_question": "Current weather conditions in New York",
     "topic_changed": true
   }
   ```

2. **Input**: "Hi, how are you?"
   **Output**: ```json
   {
     "intent": "no_retrieve",
     "reformulated_question": "",
     "topic_changed": false
   }
   ```

3. **Input**: "Tell me about Python programming" (after asking about weather)
   **Output**: ```json
   {
     "intent": "retrieve",
     "reformulated_question": "Overview of Python programming language",
     "topic_changed": true
   }
   ```

**Instructions**:
- Classify the intent as "retrieve" or "no_retrieve".
- If "retrieve," provide a concise reformulated question optimized for a retriever.
- If "no_retrieve," set reformulated_question to an empty string.
- Set topic_changed to true if the topic differs significantly from the last user input in history, false otherwise.
- Return a JSON object with keys: intent, reformulated_question, topic_changed.

**Output**:
```json
{
  "intent": "<retrieve or no_retrieve>",
  "reformulated_question": "<reformulated question or empty string>",
  "topic_changed": <true or false>
}
"""
)


# Intent Detector Node
def intent_detector_node(state: GraphState) -> GraphState:
    """Detect intent from user input and update state with the result."""
    llm = get_llm()
    prompt = INTENT_PROMPT.format(conversation_history=json.dumps(state.history, indent=2), user_input=state.user_input)
    response = llm.invoke(prompt)
    intent_output = json.loads(response.content) if isinstance(response.content, str) else response.content
    if isinstance(intent_output, dict):
        state.intent_output = intent_output
    else:
        raise ValueError("Expected response content to be a dict")
    return state


# Retriever Node
def retriever_node(state: GraphState) -> GraphState:
    """Handle retrieval intent and simulate retrieval logic."""
    reformulated_question = state.intent_output.get("reformulated_question", "")
    # Simulate retrieval logic (replace with actual retriever in production)
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
    """Compile and return the intent detection workflow graph.

    Returns:
        CompiledStateGraph: The compiled intent detection workflow graph.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("intent_detector", intent_detector_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("non_retriever", non_retriever_node)

    # Set entry point
    workflow.set_entry_point("intent_detector")

    # Add conditional edges
    workflow.add_conditional_edges(
        "intent_detector", route_intent, {"retriever": "retriever", "non_retriever": "non_retriever"}
    )

    # Add end edges
    workflow.add_edge("retriever", END)
    workflow.add_edge("non_retriever", END)

    return workflow.compile()


# Example Usage
if __name__ == "__main__":
    graph = build_graph()
    inputs = GraphState(user_input="What's the capital of France?", history=[{"user": "Hi!", "assistant": "Hello!"}])
    result = graph.invoke(inputs)
    logger.info(json.dumps(result.intent_output, indent=2))
