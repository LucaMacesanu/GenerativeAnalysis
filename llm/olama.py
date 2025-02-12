from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

local_llm = "llama3.1:8b-instruct-fp16"  # Updated model
llm = ChatOllama(model=local_llm, temperature=0)

from langchain_core.messages import AIMessage

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)