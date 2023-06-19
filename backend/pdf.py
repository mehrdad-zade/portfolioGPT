# pip install PyPDF2 transformers torch
# make sure ENV is activated
# curl https://sh.rustup.rs -sSf | bash -s -- -y --no-modify-path
# pip install transformers==2.4.1
# python 3.8
# pip install xformers


import PyPDF2

# Step 1: Convert PDFs to text
def convert_pdf_to_text(file_path):
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        text = ''
        n = len(pdf.pages)
        for page in range(n):
            text += pdf.pages[page].extract_text();
        return text


