from flask import Flask, render_template, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import os
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)

app = Flask(__name__, static_folder='templates/static')
sia = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text is empty'}), 400
        
    scores = sia.polarity_scores(text)
    
    # Determine overall sentiment
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    return jsonify({
        'sentiment': sentiment,
        'scores': scores
    })

import pandas as pd
import io

@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not (file.filename.endswith('.csv') or file.filename.endswith('.txt')):
        return jsonify({'error': 'Invalid file type. Please upload CSV or TXT.'}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            # Find the first column that contains text (roughly)
            text_col = df.select_dtypes(include=['object']).columns[0]
            texts = df[text_col].astype(str).tolist()
        else:
            texts = [line.decode('utf-8').strip() for line in file.readlines() if line.strip()]

        if not texts:
            return jsonify({'error': 'File is empty or no text found'}), 400

        total_scores = {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0}
        reviews = []
        for t in texts:
            s = sia.polarity_scores(t)
            for k in total_scores:
                total_scores[k] += s[k]
            
            # Determine individual sentiment
            if s['compound'] >= 0.05:
                ind_sentiment = 'positive'
            elif s['compound'] <= -0.05:
                ind_sentiment = 'negative'
            else:
                ind_sentiment = 'neutral'
                
            reviews.append({
                'text': t,
                'sentiment': ind_sentiment,
                'scores': s
            })
        
        count = len(texts)
        avg_scores = {k: v / count for k, v in total_scores.items()}
        
        compound = avg_scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return jsonify({
            'sentiment': sentiment,
            'scores': avg_scores,
            'count': count,
            'reviews': reviews
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
