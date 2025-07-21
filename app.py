from flask import Flask, render_template, request, jsonify
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and tokenizer using joblib
model = load_model('model.keras')
tokenizer = joblib.load('tokenizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    review = data.get('review', '')
    
    if not review.strip():
        return jsonify({'error': 'Please enter a review.'})

    # Tokenize and pad
    review_seq = tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_seq, maxlen=120, padding='post', truncating='post')

    # Predict
    prediction = model.predict(review_padded, verbose=0)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return jsonify({'sentiment': sentiment, 'score': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
