from flask import Flask, request, jsonify
from flask_cors import CORS
from upload import save_uploaded_files
from sentiment import get_sentiment


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
    file = request.files.getlist('file')[0]
    print("upload :", file)
    upload_stats = save_uploaded_files(file)
    if upload_stats == 'saved':
        file_name, sentiment = get_sentiment()
    return jsonify({'message': 'Document uploaded and parsed successfully', 'file_name': file_name, 'sentiment': sentiment})

# Endpoint for user queries
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']  # Assuming question is sent in the request body as JSON
    # return jsonify({'answer': chatGPT3_response(question)})
    return jsonify({'answer': SAMPLE_RESPONSE})

# Endpoint initial test
@app.route('/health', methods=['GET'])
def health():
    return "OK"


    
if __name__ == '__main__':
    app.run()
