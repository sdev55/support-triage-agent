from langchain_openai import ChatOpenAI

# Connect to OpenAI's hosted model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# Send a customer support issue to the model
response = llm.invoke(
    "A customer is receiving HTTP 401 after rotating "
    "their API key. What should we investigate first? "
    "Answer in 2 sentences."
)

print(response.content)