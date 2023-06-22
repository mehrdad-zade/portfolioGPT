# pip install PyPDF2 transformers torch
# make sure ENV is activated
# curl https://sh.rustup.rs -sSf | bash -s -- -y --no-modify-path
# pip install transformers==2.4.1
# python 3.8
# pip install xformers


import PyPDF2
from mySecrets import LOCAL_PATH

def convert_pdf_to_text_from_storage(file_path):
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        text = ''
        n = len(pdf.pages)
        for page in range(n):
            text += pdf.pages[page].extract_text();
        return text
    
def convert_pdf_to_text(file):
    pdf = PyPDF2.PdfReader(file)
    text = ''
    n = len(pdf.pages)
    for page in range(n):
        text += pdf.pages[page].extract_text();
    return text

def save_as_text(file):
    txt = convert_pdf_to_text(file)
    with open(LOCAL_PATH + 'data/source_of_knowledge/' + file.filename + '.txt', "w") as file:
        file.write(txt)
    return txt



