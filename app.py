from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from card import process_url_to_flashcards

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/api/generate-flashcards', methods=['POST'])
def generate_flashcards():
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        result = process_url_to_flashcards(data['url'], data.get('num_cards', 5))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)