# pip install PyPDF2 transformers torch
# make sure ENV is activated
# curl https://sh.rustup.rs -sSf | bash -s -- -y --no-modify-path
# pip install transformers==2.4.1
# python 3.8
# pip install xformers

import os
import PyPDF2
from sentiment import perform_sentiment_analysis

# Step 1: Convert PDFs to text
def convert_pdf_to_text(file_path):
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        text = ''
        n = len(pdf.pages)
        for page in range(n):
            text += pdf.pages[page].extract_text();
        return text



# Step 3: Train a question-answering model (guide provided)
# Due to the resource-intensive nature of training, this step requires separate steps and specific guidance.
# We will instead use a pre-trained model for your task.

# Main script
import os
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
pdf_folder = os.path.join(parent_directory, 'database/uploads') # Update with the folder containing your PDF files
output_table = []

for filename in os.listdir(pdf_folder):
    print(f"Processing {filename}...")
    if filename.endswith('.pdf'):        
        file_path = os.path.join(pdf_folder, filename)
        text = convert_pdf_to_text(file_path)
        sentiment = perform_sentiment_analysis(text)
        output_table.append((filename, sentiment))

# Display the output table
print("PDF File\t\tSentiment")

print("-----------------------------")
for row in output_table:
    print(f"{row[0]}\t\t{row[1]}")
