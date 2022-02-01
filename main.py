# import streamlit as st
# print(flask.__version__)
# print('sk',sk.__version__)
# print('pandas',pd.__version__)
# print('np',np.__version__)
# print('sns',sns.__version__)
# print('m',m.__version__)
# app = Flask('Heart_disease_prediction')

import flask
from flask import Flask, request, jsonify, render_template
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib as m
import seaborn as sns
import pickle

app = Flask(__name__)
model = pickle.load(open('model_files/diease_prediction_model_v1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    """
    For rendering result in HTML GUI"""
    int_features1 = [int(x) for x in request.form.values()]

    int_features = int_features1

    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    print('Prediction is ', prediction)
    if prediction:
        result = 'Have Disease'
    else:
        result = "You are safe, Don't have diease"

    output =  prediction

    return render_template('index.html', prediction_text = 'Model predict : {}'.format(result))

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    '''
    For direct api'''
    return 'In progress'

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port = 9698)

