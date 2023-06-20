import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tensorflow warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import upload
from sentiment import perform_sentiment_analysis
import pdf
from index_data import qNa_source_of_knowledge
from azure import chatGPT3_response



SAMPLE_RESPONSE = """
You can resolve CORS issue in Flask by installing and using the Flask-CORS extension. Here are the steps to do so:

1. Install Flask-CORS using pip:

```python
pip install Flask-Cors
```

2. Import the Flask-CORS extension in your Flask app:

```python
from flask_cors import CORS
```

3. Initialize the extension with your Flask app:

```python
app = Flask(__name__)
CORS(app)
```

4. Add the allowed origins to the CORS configuration:

```python
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://example.com"]}})
```

This will allow requests from the specified origins. You can also use "*" to allow any origin.

5. Change the response header to include CORS headers:

```python
@app.route("/")
def hello_world():
    return "Hello, World!", 200, {"Access-Control-Allow-Origin": "*"}
```

This will allow any origin to access the endpoint. You can also specify specific origins instead of "*" by replacing it with the desired origin.

By following these steps, you can resolve the CORS issue in your Flask app.
"""

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for all routes

# Endpoint for document upload
@app.route('/upload', methods=['POST'])
def upload_document():
    # get uploaded files: Assuming 'files' are sent in the request body as form UI
    files = request.files.getlist('files') 
    sentiment_res = {} # dictionary to store sentiment of files

    for file in files:
        # save files to database/uploads
        upload.save(file)
        # convert files to text and save them to source of knowledge
        txt = pdf.save_as_text(file)
        # get sentiment of files
        sentiment_res[file.filename] = perform_sentiment_analysis(txt)
    # index text file of source of knowledge
    # createVectorIndex()

    # return jsonify({'message': 'Document uploaded and parsed successfully', 'file_name': file_name, 'sentiment': sentiment})
    return jsonify({'message': 'Document uploaded and parsed successfully', 'sentiment': sentiment_res})

# Endpoint for user queries
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']  # Assuming question is sent in the request body as JSON
    answer = chatGPT3_response(question)
    if answer == None or 'September 2021' in answer or 'As an AI language model' in answer:
        pre_text = "I'm sorry I couldn't find the answer to your question in the public chatGPT domain. However, I can assist you with your private data.\n"
        answer = pre_text + str(qNa_source_of_knowledge(question))
    return jsonify({'answer': answer})

# Endpoint initial test
@app.route('/health', methods=['GET'])
def health():
    return "OK"


    
if __name__ == '__main__':
    app.run()


'''
pipeline:

- upload files:
    - save files to database/uploads
    - parse files to text
    - index text as source of knowledge

- ask questions:
    - get question from user
    - query source of knowledge


'''