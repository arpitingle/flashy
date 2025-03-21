# Flashy: AI Flashcard Generator 

A web application that automatically generates interactive flashcards from articles and YouTube videos using AI.

## ✨ Features

- **Article Processing**: Convert blog posts/articles to flashcards
- **YouTube Support**: Create flashcards from video transcripts
- **Responsive Design**: Grid-based card layout that works on mobile/desktop
- **Flip Animation**: Smooth card-flip transition for Q/A
- **Customizable Count**: Choose number of flashcards (1-10)
- **Error Handling**: Clear messages for invalid URLs/API failures

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/arpitingle/flashy.git
cd flashy
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Config**

- Create .env File
```bash
# Required
OPENROUTER_API_KEY=your_openrouter_key_here
``` 

- Get API Key
Create free account at OpenRouter.ai
Navigate to Keys Page
Generate new key and paste into .env

3. Start 

```bash
python app.py
```

![Flashcard Demo](https://github.com/arpitingle/flashy/blob/main/demo.png) 



