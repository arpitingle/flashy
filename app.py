from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from card import process_url_to_flashcards

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/generate-flashcards', methods=['GET', 'POST'])
def generate_flashcards():
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({"error": "URL is required"}), 400
    
    url = data['url']
    num_cards = data.get('num_cards', 5)  # Default to 5 cards if not specified
    
    try:
        result = process_url_to_flashcards(url, num_cards)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)