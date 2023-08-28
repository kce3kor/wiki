import os
import openai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]


llm_model = "gpt-3.5-turbo-0301"

def get_completion(prompt, model=llm_model):
    message = [{"role":"user", "content":prompt}]

    response = openai.ChatCompletion.create(
        model = model,
        messages = message,
        temperature = 0,
    )
    return response.choices[0].message["content"]


print(get_completion("What is 1+1?"))

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """British English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)

print(get_completion(prompt))