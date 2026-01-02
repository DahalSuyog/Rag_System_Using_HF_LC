import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.tools import tool
from utils import search_university_data
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_id = "google/functiongemma-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")



pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

def agentho(query: str):
    """Create and run an agent to answer the user's query using the university data."""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Search the Excel database for German university fees and info."""
        retrieved_docs = search_university_data(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs



    tools = [retrieve_context]

    # If desired, specify custom instructions
    prompt= """<start_of_turn>developer
    You are a university assistant. Answer the user's question using the tools provided.
    Tools available: {tools}
    Tool Names: {tool_names}

    To use a tool, follow this EXACT format:
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!<end_of_turn>
    <start_of_turn>user
    {input}
    {agent_scratchpad}<end_of_turn>
    <start_of_turn>model
    """

    agent = create_agent(chat_model, tools, system_prompt=prompt)

    messages = [{"role": "user", "content": query}]

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()