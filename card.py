import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set up OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "https://your-app-name.com")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "AI Flashcard Generator")

# Initialize OpenAI client with OpenRouter base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ContentProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_content_from_url(self, url):
         """Extract main content from a given URL"""
         try:
             headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        # Handle potential SSL verification issues
             response = requests.get(url, headers=headers, timeout=10, verify=False)
             response.raise_for_status()
        
             soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove unnecessary elements
             for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'form', 'button', 'iframe', 'noscript']):
                element.decompose()

        # Try to find article content using multiple strategies
             article = soup.find('article') or soup.find('main') or soup.find('div', role='main')
        
             if article:
                    main_content = article.get_text(separator='\n', strip=True)
             else:
            # Fallback to body content with paragraph filtering
                 body = soup.find('body')
                 paragraphs = body.find_all(['p', 'h1', 'h2', 'h3']) if body else []
                 main_content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        # Clean up excessive whitespace
             main_content = re.sub(r'\n{3,}', '\n\n', main_content.strip())
        
             return {
            "title": soup.title.string.strip() if soup.title else "Untitled",
            "content": main_content,
            "url": url
        }
        
         except Exception as e:
          return {"error": f"Error processing URL: {str(e)}", "url": url}
    
    def extract_key_sentences(self, content, max_sentences=20):
        """Extract important sentences from the content"""
        sentences = sent_tokenize(content)
        if not sentences:
            return []
        
        word_freq = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if word not in self.stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words if word not in self.stop_words)
            if len(words) > 0:
                score /= len(words)
            sentence_scores[i] = score
        
        top_indices = sorted(sentence_scores.keys(), key=lambda i: sentence_scores[i], reverse=True)[:max_sentences]
        top_indices.sort()
        
        return [sentences[idx] for idx in top_indices]

class FlashcardGenerator:
    def generate_flashcards_via_deepseek(self, content_data, num_cards=5):
        """Generate flashcards using OpenRouter's Deepseek R1 model"""
        title = content_data["title"]
        important_sentences = content_data["key_sentences"]
        content_summary = " ".join(important_sentences)
        
        system_message = """
        You are an educational content expert. Your task is to create effective flashcards (question-answer pairs) 
        based on the provided text. Follow these guidelines:
        
        1. Create concise, clear questions that test understanding, not just memorization
        2. Make answers brief but complete
        3. Focus on the most important concepts
        4. Avoid overly complex questions
        5. Make sure questions and answers are directly based on the provided content
        
        Output ONLY valid JSON in the format:
        [
            {"question": "Question text here", "answer": "Answer text here"},
            ...
        ]
        """
        
        user_message = f"""
        Title: {title}
        
        Content:
        {content_summary}
        
        Please generate {num_cards} high-quality flashcards based on this content.
        """
        
        try:
            response = client.chat.completions.create(
                extra_headers={"HTTP-Referer": YOUR_SITE_URL, "X-Title": YOUR_SITE_NAME},
                model="deepseek/deepseek-r1:free",
                messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                temperature=0.3,
                max_tokens=1000
            )
            
            flashcards_json = response.choices[0].message.content.strip()
            flashcards_json = re.sub(r'^```json', '', flashcards_json)
            flashcards_json = re.sub(r'```$', '', flashcards_json).strip()
            return json.loads(flashcards_json)
        except Exception as e:
            return [{"question": "API Error", "answer": f"Error communicating with AI service: {str(e)}"}]

def process_url_to_flashcards(url, num_flashcards=5):
    content_processor = ContentProcessor()
    flashcard_generator = FlashcardGenerator()
    content_data = content_processor.extract_content_from_url(url)
    
    if "error" in content_data:
        return {"error": content_data["error"], "title": "Error", "url": url, "flashcards": []}
    
    key_sentences = content_processor.extract_key_sentences(content_data["content"])
    if not key_sentences:
        return {"error": "Could not extract meaningful content", "title": content_data["title"], "url": url, "flashcards": []}
    
    content_data["key_sentences"] = key_sentences
    flashcards = flashcard_generator.generate_flashcards_via_deepseek(content_data, num_flashcards)
    
    return {"title": content_data["title"], "url": content_data["url"], "flashcards": flashcards}