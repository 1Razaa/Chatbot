from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('cb_model')

# Load the tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

keras.backend.set_learning_phase(0)

@app.route('/')
def home():
    return render_template('index.html', chats=[])


@app.route('/chat', methods=['POST'])
def predict():
    prompt = request.form['prompt']

    # Perform the same prediction steps as in the chat() function
    max_len = 20
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([prompt]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]  # Convert to scalar value

    # Load the intents from the JSON file
    with open('intents.json') as file:
        data = json.load(file)

    for intent in data['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            return render_template('index.html', chats=[{'user': True, 'message': prompt},
                                                       {'user': False, 'message': response}])

    return render_template('index.html', chats=[{'user': True, 'message': prompt},
                                                {'user': False, 'message': 'Sorry, I did not understand that.'}])

if __name__ == '__main__':
    app.run()
