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
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Extract title
            title = soup.title.string if soup.title else "Untitled"
            
            # Extract main content (this is a simple approach and might need refinement)
            # Looking for main content in article tags, main tag, or div with content-related classes
            main_content = ""
            
            # Try to find article or main content
            article_tag = soup.find('article') or soup.find('main')
            if article_tag:
                main_content = article_tag.get_text(separator=' ', strip=True)
            else:
                # Look for common content div patterns
                content_divs = soup.find_all('div', class_=re.compile(r'(content|article|post|entry)'))
                if content_divs:
                    for div in content_divs:
                        main_content += div.get_text(separator=' ', strip=True) + " "
                else:
                    # Fallback to body text, excluding typical non-content areas
                    body = soup.find('body')
                    if body:
                        main_content = body.get_text(separator=' ', strip=True)
            
            # Clean up the text
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            
            return {
                "title": title,
                "content": main_content,
                "url": url
            }
            
        except Exception as e:
            return {
                "error": f"Error processing URL: {str(e)}",
                "url": url
            }
    
    def extract_key_sentences(self, content, max_sentences=20):
        """Extract important sentences from the content"""
        # Break content into sentences
        sentences = sent_tokenize(content)
        
        if not sentences:
            return []
        
        # Simple approach: extract sentences with important keywords
        # In a more advanced version, you might use TextRank or similar algorithms
        important_sentences = []
        
        # Calculate word frequency excluding stop words
        word_freq = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if word not in self.stop_words and len(word) > 3:
                    if word not in word_freq:
                        word_freq[word] = 1
                    else:
                        word_freq[word] += 1
        
        # Score sentences based on important words
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words if word not in self.stop_words)
            # Normalize by sentence length to avoid bias towards longer sentences
            if len(words) > 0:
                score = score / len(words)
            sentence_scores[i] = score
        
        # Select top sentences
        top_indices = sorted(sentence_scores.keys(), key=lambda i: sentence_scores[i], reverse=True)[:max_sentences]
        top_indices = sorted(top_indices)  # Maintain original order
        
        for idx in top_indices:
            important_sentences.append(sentences[idx])
        
        return important_sentences


class FlashcardGenerator:
    def generate_flashcards_via_deepseek(self, content_data, num_cards=5):
        """Generate flashcards using OpenRouter's Deepseek R1 model"""
        title = content_data["title"]
        important_sentences = content_data["key_sentences"]
        
        # Prepare content for the prompt
        content_summary = " ".join(important_sentences)
        
        # Prepare the prompt
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
            {
                "question": "Question text here",
                "answer": "Answer text here"
            },
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
                extra_headers={
                    "HTTP-Referer": YOUR_SITE_URL,
                    "X-Title": YOUR_SITE_NAME,
                },
                model="deepseek/deepseek-r1:free",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract and parse the JSON response
            flashcards_json = response.choices[0].message.content.strip()
            
            # Handle potential formatting issues
            try:
                # Clean up the response to handle cases where the JSON might be wrapped in code blocks
                flashcards_json = re.sub(r'^```json', '', flashcards_json)
                flashcards_json = re.sub(r'```$', '', flashcards_json)
                flashcards_json = flashcards_json.strip()
                
                flashcards = json.loads(flashcards_json)
                return flashcards
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the JSON part
                json_pattern = r'\[\s*\{.*\}\s*\]'
                match = re.search(json_pattern, flashcards_json, re.DOTALL)
                if match:
                    try:
                        flashcards = json.loads(match.group(0))
                        return flashcards
                    except:
                        return [{"question": "Error parsing AI response", "answer": "The AI generated invalid JSON. Please try again."}]
                else:
                    return [{"question": "Error parsing AI response", "answer": "The AI generated invalid JSON. Please try again."}]
                
        except Exception as e:
            return [{"question": "API Error", "answer": f"Error communicating with AI service: {str(e)}"}]


def process_url_to_flashcards(url, num_flashcards=5):
    """Main function to process a URL and generate flashcards"""
    # Initialize the processors
    content_processor = ContentProcessor()
    flashcard_generator = FlashcardGenerator()
    
    # Extract content
    content_data = content_processor.extract_content_from_url(url)
    
    if "error" in content_data:
        return {"error": content_data["error"], "title": "Error", "url": url, "flashcards": []}
    
    # Extract key sentences
    key_sentences = content_processor.extract_key_sentences(content_data["content"])
    
    # Check if we have enough content
    if not key_sentences:
        return {"error": "Could not extract meaningful content from the URL", 
                "title": content_data["title"], 
                "url": url, 
                "flashcards": []}
    
    content_data["key_sentences"] = key_sentences
    
    # Generate flashcards
    flashcards = flashcard_generator.generate_flashcards_via_deepseek(content_data, num_flashcards)
    
    return {
        "title": content_data["title"],
        "url": content_data["url"],
        "flashcards": flashcards
    }


# Example usage
if __name__ == "__main__":
    # Example URL - replace with any article URL
    url = "https://example.com/article"
    result = process_url_to_flashcards(url, 5)
    
    # Print the result in a readable format
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print("\nFlashcards:")
    for i, card in enumerate(result['flashcards'], 1):
        print(f"\nCard {i}:")
        print(f"Q: {card['question']}")
        print(f"A: {card['answer']}")