import flask
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def model(arg):
    df=pd.read_csv(arg)
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
    meta = model("input1.csv")
    lymph = model("input2.csv")

    result1 = meta.predict(to_predict)
    result2 = lymph.predict(to_predict)
    return result1[0], result2[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        pr = to_predict_list[0]
        to_predict_list = list(map(float, to_predict_list[1:]))
        result1, result2 = ValuePredictor(to_predict_list)

        if int(result1) == 1:
            prediction1 = 'Metastasis'
        else:
            prediction1 = 'Non Metastasis'
        if int(result2) == 1:
            prediction2 = 'lymph'
        else:
            prediction2 = 'Non lymph'

        return render_template("result.html", prediction1=prediction1,prediction2=prediction2,mrno=pr)

