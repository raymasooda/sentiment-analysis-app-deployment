from flask import Flask, request, jsonify, render_template
import pandas as pd

import joblib
import utils as u
from utils import preprocessor

app = Flask(__name__)
model = joblib.load(open('model.joblib','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    input = request.form['text']
    processed_input = pd.Series(u.convert_text(u.clean_text(input)))
    predicted_sentiment = model.predict(processed_input)
    if predicted_sentiment == 1:
        output = 'positive'
    else:
        output = 'negative'

    return render_template('index.html', sentiment=f'Predicted sentiment of "{input}" is {output}.')


if __name__ == "__main__":
    app.run(debug=True)