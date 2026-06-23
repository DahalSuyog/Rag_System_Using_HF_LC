
from __future__ import annotations
 
import os
from typing import Any
from dotenv import load_dotenv
 
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from .tools import get_all_tools

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError(
        "HF_TOKEN not found in environment variables. Please set it in your .env file or pass it directly to the Model constructor."
    )

TOOLS = get_all_tools()

class Model:
    """
    Wraps a Langchain.agents agent backed by Llama 3 8B on HuggingFace.
 
    Usage
    -----
        model = Model(hf_token="hf_...")
        response = model.run("What does the doc say about transformers?")
 
        # With conversation history
        history = [HumanMessage("hi"), AIMessage("Hello! How can I help?")]
        response = model.run("Tell me about attention mechanisms", history=history)
 
    Parameters
    ----------
    hf_token : str
        HuggingFace API token. Defaults to env var HF_TOKEN if not passed.
    model_id : str
        HF model repo id. Must be a chat/instruct model.
    temperature : float
        0.0 to 0.2 recommended for reliable tool-call JSON generation.
    max_new_tokens : int
        Per-step generation budget.
    """
    def __init__(self,
        hf_token: str | None = None,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        tools = None
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "Provide hf_token= or set the HF_TOKEN environment variable. "
                "Get a token at https://huggingface.co/settings/tokens"
            )
 
        if tools is None:
            tools = get_all_tools()
 
        # HuggingFaceEndpoint: calls the HF Serverless Inference API.
        # For a private/dedicated endpoint swap `repo_id` for `endpoint_url`.
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
 
        # ChatHuggingFace adds the chat template and bind_tools support
        # that the raw HuggingFaceEndpoint lacks.
        chat_model = ChatHuggingFace(llm=llm)
 
        system_prompt = (
            "You are a helpful AI assistant with access to tools. "
            "Use the greet_user tool when the user is greeting you. "
            "Use the retrieve_documents tool when the user asks a question "
            "that requires looking up information. "
            "For all other messages, respond directly."
        )
 
        # create_react_agent compiles the full ReAct graph:
        #   [call_model] → (tool call?) → [tool_node] → [call_model] → …
        # It handles: tool dispatch, ToolMessage injection, loop termination.
        self.agent = create_agent(
            model=chat_model,
            tools=TOOLS,
            system_prompt=system_prompt,
        )


    def run(self, user_message: str, history: list[BaseMessage] | None = None,
    ) -> str:
        """
        Run the agent on a user message and return the final text response.
 
        Parameters
        ----------
        user_message : str
            The latest message from the user.
        history : list[BaseMessage], optional
            Prior conversation turns. Pass alternating HumanMessage /
            AIMessage objects.
 
        Returns
        -------
        str
            The agent's final natural-language response.
        """
        messages: list[BaseMessage] = list(history or [])
        messages.append(HumanMessage(content=user_message))
 
        result = self.agent.invoke({"messages": messages})
 
        # result["messages"] is the full updated message list.
        # The last AIMessage without tool_calls is the final answer.
        final = self._extract_final_response(result["messages"])
        return final
 
    @staticmethod
    def _extract_final_response(messages: list[BaseMessage]) -> str:
        """
        Walk the message list in reverse and return the last AIMessage
        that is NOT a tool call (i.e. the actual response to the user).
        """
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                # tool_calls present → intermediate step, skip it
                if getattr(msg, "tool_calls", None):
                    continue
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return ""
    
if __name__ == "__main__":
    
    aimodel = Model()   # reads HF_TOKEN from env
 
    turns = [
        "Hey there!",
        "Hi, my name is Sita.",
        "What does the knowledge base say about neural networks?",
    ]

    history: list[BaseMessage] = []
    for user_input in turns:
        print(f"\nUSER: {user_input}")
        response = aimodel.run(user_input, history=history)
        print(f"ASSISTANT: {response}")
        # accumulate history for multi-turn
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))