from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def front():
    return render_template("template.html")

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']

    arr = np.array([data1, data2, data3, data4, data5, data6, data7, data8, data9])

    filename = 'encoders.sav'
    encoders = pickle.load(open(filename, 'rb'))
    data = arr.tolist()
    #data =  ['22', 'female', '1 - unskilled and resident', 'own', 'little', 'moderate', '2000', '12', 'business']
    try:
        data = input_encoding(data, encoders)
    except:
        data =  [  34,    1,    2,    0,    0,    0, 6527,   60,    1] #bad
        #data = [22, 0, 2, 1, 0, 1, 5951, 48, 5] #good

    #model
    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict([data])
    data.append(int(result))

    return render_template("output.html", data=data)

def input_encoding(sample, encoders):
    sample[0] = int(sample[0])
    sample[1] = int(encoders["Sex"].transform([sample[1]]))
    sample[2] = int(encoders["Job"].transform([sample[2]]))
    sample[3] = int(encoders["Housing"].transform([sample[3]]))
    sample[4] = int(encoders["Saving_accounts"].transform([sample[4]]))
    sample[5] = int(encoders["Checking_account"].transform([sample[5]]))
    sample[6] = int(sample[6])
    sample[7] = int(sample[7])
    sample[8] = int(encoders["Purpose"].transform([sample[8]]))

    return sample


if __name__ == "__main__":
    app.run(debug=True)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"