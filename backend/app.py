from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Endpoint for document upload
@app.route('/upload', methods=['POST'])
def upload_document():
    file = request.files['file']  # Assuming file input field name is 'file'
    # Save the file to a desired location
    file.save('path/to/save/' + file.filename)
    
    # Call function for document parsing
    parsed_data = parse_document('path/to/save/' + file.filename)
    
    #return jsonify({'message': 'Document uploaded and parsed successfully', 'parsed_data': parsed_data})
    print("upload_document")
    return jsonify({'message': 'Document uploaded and parsed successfully', 'parsed_data': 'TBD'})

# Endpoint for user queries
@app.route('/ask', methods=['POST'])
def ask_question():
    # question = request.json['question']  # Assuming question is sent in the request body as JSON
    # response = query_chatgpt(question)
    # return jsonify({'response': response})
    print("ask_question")
    return jsonify({'response': 'TBD'})

# Endpoint initial test
@app.route('/health', methods=['GET'])
def health():
    return "OK"

# Function to parse the document
def parse_document(file_path):
    # Implement your document parsing logic here
    # Extract relevant information from the document and return it
    parsed_data = {
        'title': 'Sample Document',
        'author': 'John Doe',
        'content': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
    }
    return parsed_data

# Function to query the ChatGPT API
def query_chatgpt(question):
    # Implement your ChatGPT API query logic here
    # Make a POST request to the ChatGPT API and return the response
    api_url = 'https://api.openai.com/v1/engines/davinci-codex/completions'
    headers = {'Authorization': 'Bearer YOUR_API_KEY'}
    payload = {'prompt': question, 'max_tokens': 50}
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()['choices'][0]['text']
    
if __name__ == '__main__':
    app.run()
