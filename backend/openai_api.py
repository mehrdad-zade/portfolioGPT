'''
pip install gpt_index
pip install langchain
pip install transformers
'''

import os
import openai
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from mySecrets import LOCAL_PATH, OpenAI_API_KEY_PERSONAL

os.environ['OPENAI_API_KEY'] = OpenAI_API_KEY_PERSONAL
openai.organization = "org-8kRclvZ4TrPB1yhir9Eqn6DJ"
vIdx = LOCAL_PATH + 'database/source_of_knowledge/vectorIndex.json'


def createVectorIndex():

    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    #define LLM
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_token=tokens)) #text-davinci-003

    #load data
    docs = SimpleDirectoryReader(LOCAL_PATH + 'database/source_of_knowledge').load_data()

        
    vectorIndex = GPTSimpleVectorIndex(documents=docs, llm_predictor=llmPredictor, prompt_helper=prompt_helper)
    vectorIndex.save_to_disk(vIdx)

    return vectorIndex

def qNa():
    if os.path.isfile(vIdx):
        vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
        while True:
            prompt = input('Please ask your question here: ')
            if prompt.lower() != "goodbye.":
                response = vIndex.query(prompt, response_mode="compact")
                print(f"Response: {response} \n")
            else:
                print("Bot:- Goodbye!")
                break
    return "No source of knowledge found. Please upload documents first."

# vectorIndex = createVectorIndex('source_of_knowledge/data')
# qNa()

def qNa_source_of_knowledge(question):
    os.environ['OPENAI_API_KEY'] = OpenAI_API_KEY_PERSONAL
    openai.organization = "org-8kRclvZ4TrPB1yhir9Eqn6DJ"
    if os.path.isfile(vIdx):
        vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
        print('2------------------------------------------')
        response = vIndex.query(question, response_mode="compact")
        print('3------------------------------------------')
        return response
    print('4------------------------------------------')
    return "No source of knowledge found. Please upload documents first."

# # example
# question = "what do you know about onex" 
# print(qNa_source_of_knowledge(question))


def chatGPT3_response(user_input):
    openai.api_key = OpenAI_API_KEY_PERSONAL
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": user_input}]
        )
    return res["choices"][0]["message"]["content"]

# # Example usage
# prompt = "What is the capital of France?"
# response = chatGPT3_response(prompt)
# print(response)