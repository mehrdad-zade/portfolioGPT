from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for all routes

# Endpoint for document upload
@app.route('/upload', methods=['POST'])
def upload_document():
    file = request.files['file']  # Assuming file input field name is 'file'
    # Save the file to a desired location
    # file.save('path/to/save/' + file.filename)
    
    # Call function for document parsing
    parsed_data = parse_document()
    return jsonify({'message': 'Document uploaded and parsed successfully', 'parsed_data': 'parsed_data'})

# Endpoint for user queries
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']  # Assuming question is sent in the request body as JSON
    # response = query_chatgpt(question)
    # return jsonify({'response': response})
    print("ask_question")
    return jsonify({'answer': 'TBD - ' + question})

# Endpoint initial test
@app.route('/health', methods=['GET'])
def health():
    return "OK"


    
if __name__ == '__main__':
    app.run()
