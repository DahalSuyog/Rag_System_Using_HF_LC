# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.tools import tool
from utils import search_university_data
from langchain.agents import create_agent


model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

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
    prompt= """You are a university assistant.
You have access to the following tools:
{tools}

To use a tool, follow this EXACT format:
Thought: I need to check the database for fee information.
Action: university_search_tool
Action Input: {input}
Observation: [Tool will return data here]

Final Answer: [Your summary of the data]

Begin!
Question: {input}
{agent_scratchpad}"""

    agent = create_agent(chat_model, tools, system_prompt=prompt)

    messages = [{"role": "user", "content": query}]

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()