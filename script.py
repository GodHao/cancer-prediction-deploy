import flask
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def model():
    df=pd.read_csv("input.csv")
    inter = np.array(df)
    features=inter[:,:7]
    inter3=inter[:,7:8]
    label=[]
    for i in inter3:
      for j in i:
          label.append(int(j))
    X_train = features[:139]
    Y_train = label[:139]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,Y_train)
    return neigh

app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 7)
    print(to_predict)
    loaded_model = model()
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)

        if int(result) == 1:
            prediction = 'Metastasis'
        else:
            prediction = 'Non Metastasis'

        return render_template("result.html", prediction=prediction)

