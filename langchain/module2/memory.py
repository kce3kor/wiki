"""
Conversational Buffer Memory and Conversational Chain



- LLM are stateless which means each API call is independent , so how does chat history work?
- Chatbots, LLMs appear to have memory by providing the full conversation as "context"
- Conversational Buffer Memory is a stateful object which maintains a history of conversations
- Providing the entire conversation (large number of tokens), can be expensive as the costings are based on the number of tokens so
- So, Langchain provides several ways to accumulate and process the memory conversations eg: Conversational BufferWindowMemory
- Memory buffer is an important concept

"""


import os
import openai
from dotenv import load_dotenv, find_dotenv

import warnings
warnings.filterwarnings('ignore')

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

llm_model = "gpt-3.5-turbo-0301"
##############################################

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

print(conversation.predict(input="Hi, my name is Andrew"))
print(conversation.predict(input="What is 1+1?"))
print(conversation.predict(input="What is my name?"))


print(memory.buffer)


print(memory.load_memory_variables({}))

memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})

print(memory.load_memory_variables({}))

