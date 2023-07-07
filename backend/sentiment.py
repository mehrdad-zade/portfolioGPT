import os

from pdf import convert_pdf_to_text



# training a model: https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/

def perform_sentiment_analysis(text):
    from transformers import pipeline 
    classifier = pipeline("sentiment-analysis")
    sentiment = classifier(text)[0]
    return sentiment['label']

def get_sentiment():   
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    pdf_folder = os.path.join(parent_directory, 'data/uploads') # Update with the folder containing your PDF files
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
    for row in output_table:
        print(f"{row[0]}\t\t{row[1]}")
    # return output_table[1][0], output_table[1][1]

def sentiment_polarity(txt):
    from textblob import TextBlob
    content = ''
    file = open('call_centre_sample.txt', "r")
    content = file.read()
    blob = TextBlob(content)
    sentiment = blob.sentiment.polarity
    return sentiment




# get_sentiment()
# print(perform_sentiment_analysis("data/source_of_knowledge/call_centre_sample.txt"))
print(sentiment_polarity('data/source_of_knowledge/call_centre_sample.txt'))