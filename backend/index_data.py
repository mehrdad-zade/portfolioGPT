'''
pip install gpt_index
pip install langchain
pip install transformers
'''

import os
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from mySecrets import OpenAI_API_KEY
from mySecrets import LOCAL_PATH

vIdx = LOCAL_PATH + 'database/source_of_knowledge/vectorIndex.json'
os.environ["OPENAI_API_KEY"] = OpenAI_API_KEY

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
    vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
    while True:
        prompt = input('Please ask your question here: ')
        if prompt.lower() != "goodbye.":
            response = vIndex.query(prompt, response_mode="compact")
            print(f"Response: {response} \n")
        else:
            print("Bot:- Goodbye!")
            break

def qNa_source_of_knowledge(question):
    if os.path.isfile(vIdx):
        vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
        response = vIndex.query(question, response_mode="compact")
        return response
    return "No source of knowledge found. Please upload documents first."

# vectorIndex = createVectorIndex('source_of_knowledge/data')
# qNa()s