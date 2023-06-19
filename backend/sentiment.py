import os
from transformers import pipeline
from pdf import convert_pdf_to_text


def perform_sentiment_analysis(text):
    classifier = pipeline("sentiment-analysis")
    sentiment = classifier(text)[0]
    return sentiment['label']

def get_sentiment():   
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    pdf_folder = os.path.join(parent_directory, 'database/uploads') # Update with the folder containing your PDF files
    output_table = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):        
            file_path = os.path.join(pdf_folder, filename)
            text = convert_pdf_to_text(file_path)
            sentiment = perform_sentiment_analysis(text)
            output_table.append((filename, sentiment))

    # Display the output table
    print("PDF File\t\tSentiment")

    print("-----------------------------")
    print(output_table[0][0], output_table[0][1])
    # for row in output_table:
    #     print(f"{row[0]}\t\t{row[1]}")
    return output_table[1][0], output_table[1][1]

# get_sentiment()