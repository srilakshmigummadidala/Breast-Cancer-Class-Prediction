import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib
import pickle

# create flask app
app = Flask(__name__)

# load the pickle model
model = pickle.load(open('rf_classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Clump_Thickness = request.form['Clump_Thickness']
    Uniformity_of_Cell_Size = request.form['Uniformity_of_Cell_Size']
    Uniformity_of_Cell_Shape = request.form['Uniformity_of_Cell_Shape']
    Marginal_Adhesion = request.form['Marginal_Adhesion']
    Single_Epithelial_Cell_Size = request.form['Single_Epithelial_Cell_Size']
    Bare_Nuclei = request.form['Bare_Nuclei']
    Bland_Chromatin = request.form['Bland_Chromatin']
    Normal_Nucleoli = request.form['Normal_Nucleoli']
    Mitoses = request.form['Mitoses']

    df = pd.DataFrame({'Clump_Thickness': [Clump_Thickness],'Uniformity_of_Cell_Size': [Uniformity_of_Cell_Size],
                       'Uniformity_of_Cell_Shape': [Uniformity_of_Cell_Shape],'Marginal_Adhesion': [Marginal_Adhesion],
                       'Single_Epithelial_Cell_Size':[Single_Epithelial_Cell_Size],'Bare_Nuclei': [Bare_Nuclei],
                       'Bland_Chromatin': [Bland_Chromatin],'Normal_Nucleoli': [Normal_Nucleoli],'Mitoses':[Mitoses]})


    prediction = model.predict(df)

    label_mapping = {
        2: "Benign - (non-cancerous).",
        4: "Malignant - (cancerous)) lump in a breast.",
    }

    predicted_label = label_mapping.get(prediction[0], "Unknown")

    print(f"Prediction of Breast Cancer is {predicted_label}")

    return render_template('index.html', prediction_text='Prediction of Breast Cancer Class is {}'.format(predicted_label))

if __name__ == '__main__':
    app.run(debug=True)
