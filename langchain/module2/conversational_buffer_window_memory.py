"""
Conversational Buffer Window Memory

- Only keep a window of the memory specified by k
- k=1 means only keep one response from the AI and one from the human 

"""


import os
import openai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

llm_model = "gpt-3.5-turbo-0301"
##############################################

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)


# USE OF THE VALUE k=1
memory = ConversationBufferWindowMemory(k=1)               
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

print(memory.load_memory_variables({}))


llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)
print(conversation.predict(input="Hi, my name is Andrew"))
print(conversation.predict(input="What is 1+1?"))

print(conversation.predict(input="What is my name?"))
# For this the LLM will say that it has no information of the name as the value of k=1

"""
Some other examples:
-----------------

- Conversational Token Memory: Limit the number of tokens taken for the context

    from langchain.memory import ConversationTokenBufferMemory
    from langchain.llms import OpenAI

    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
    memory.save_context({"input": "AI is what?!"},
                        {"output": "Amazing!"})
    memory.save_context({"input": "Backpropagation is what?"},
                        {"output": "Beautiful!"})
    memory.save_context({"input": "Chatbots are what?"}, 
                        {"output": "Charming!"})

    print(memory.load_memory_variables({}))


- ConversationSummaryMemory: Use an LLM to write the summary of the conversation so far, set the limit in number of tokens there

    from langchain.memory import ConversationSummaryBufferMemory

    # create a long string
    schedule = "There is a meeting at 8am with your product team. \
    You will need your powerpoint presentation prepared. \
    9am-12pm have time to work on your LangChain \
    project which will go quickly because Langchain is such a powerful tool. \
    At Noon, lunch at the italian resturant with a customer who is driving \
    from over an hour away to meet you to understand the latest in AI. \
    Be sure to bring your laptop to show the latest LLM demo."

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},
                        {"output": "Cool"})
    memory.save_context({"input": "What is on the schedule today?"}, 
                        {"output": f"{schedule}"})

    
    print(memory.load_memory_variables({}))

    conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
    )
    print(conversation.predict(input="What would be a good demo to show?"))

    print(memory.load_memory_variables({}))


"""