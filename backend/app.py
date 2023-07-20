import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # ignore tensorflow warnings. make surer this line is before importting tensorflow
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from sentiment import perform_sentiment_analysis
from openai_api import qNa_source_of_knowledge, createVectorIndex, chatGPT3_response
from validation import gpt_res_is_invalid
from prediction  import getPredictionPlot
# from custom_azure import chatGPT3_response
import upload
import pdf
import webbrowser

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for all routes
# CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5000"}})  # Enable CORS for all routes
CORS(app)

# Endpoint for document upload
@app.route('/upload', methods=['POST'])
def upload_document():
    # get uploaded files: Assuming 'files' are sent in the request body as form UI
    files = request.files.getlist('files') 
    sentiment_res = {} # dictionary to store sentiment of files

    for file in files:
        # save files to data/uploads
        upload.save(file)
        # convert files to text and save them to source of knowledge
        txt = pdf.save_as_text(file)
        # get sentiment of files
        # sentiment_res[file.filename] = perform_sentiment_analysis(txt)
    # index text file of source of knowledge
    createVectorIndex()

    return jsonify({'message': 'Document uploaded and parsed successfully', 'sentiment': sentiment_res})

# Endpoint for user queries
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']  # Assuming question is sent in the request body as JSON

    if "get stock prediction" in question:
        getPredictionPlot()
        answer = "Please find the stock prediction plot"
        hasImage = True
        return jsonify({'answer': answer, 'hasImage': hasImage})

    answer = chatGPT3_response(question)
    if gpt_res_is_invalid(answer):
        pre_text = "I'm sorry I couldn't find the answer to your question in the public chatGPT domain. However, I can assist you with your private data.\n"        
        answer = pre_text + str(qNa_source_of_knowledge(question))        
    return jsonify({'answer': answer, 'hasImage': False})

@app.route('/plot', methods=['GET'])
def show_image():
    # Render the image using the plot.html template
    return render_template('plot.html')


# Endpoint initial test
@app.route('/health', methods=['GET'])
def health():
    return "OK"


    
if __name__ == '__main__':
    app.run()


'''
pipeline:

- upload files:
    - save files to data/uploads
    - parse files to text
        - index text as source of knowledge
        - get sentiment of files
            - update sentiment table on database

- ask questions:
    - get question from user
    - if the gpt response doesn't have public info about your question
        - query source of knowledge
    - else:
        provide gpt response
    
presentation:

- performance
- storage


issues:
- cannot plot on :5000/plot
- even /health gets corropted


'''