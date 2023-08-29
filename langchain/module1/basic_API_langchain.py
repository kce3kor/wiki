"""
Basic OpenAI API call to chat llm to get the response

Source: https://learn.deeplearning.ai/langchain/lesson/2/models,-prompts-and-parsers

Prompt ---> Models ---> Response




"""

import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

llm_model = "gpt-3.5-turbo-0301"

chat = ChatOpenAI(temperature=0.0, model=llm_model)

print(chat)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

print(prompt_template.messages[0].prompt)
print(prompt_template.messages[0].prompt.input_variables)

style = """American English \
in a calm and respectful tone
"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(style=style, text=customer_email)

print(type(customer_messages))
print(type(customer_messages[0]))
print(customer_messages[0])

customer_response = chat(customer_messages)
print(customer_response.content)


service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""
service_messages = prompt_template.format_messages(
    style=service_style_pirate, text=service_reply
)
print(service_messages[0].content)

service_response = chat(service_messages)
print(service_response.content)
