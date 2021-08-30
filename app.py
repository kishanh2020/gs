from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin
from wsgiref import simple_server


app = Flask(__name__)
CORS(app)
model = pickle.load(open('mlr.sav', 'rb'))

@app.route('/')
def home():
    return render_template('home1.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()

    if request.method == 'POST':
        ballpark_id = request.form['ballpark_id']
        if ballpark_id == 'NYC20':
            temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'STP01':
            temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'MIL06':
            temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'CHI11':
            temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'LOS03':
            temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'STL10':
            temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'SFO03':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'NYC21':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'SEA03':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'PHO06':
            temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif ballpark_id == 'TOR02':
            temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif ballpark_id == 'DEN02':
            temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif ballpark_id == 'ARL02':
            temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif ballpark_id == 'WAS11':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif ballpark_id == 'CHI12':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
        elif ballpark_id == 'DET05':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
            temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        
        try:     
            H_off_avg = float(request.form['H-off_avg'])
            H_off_ops = float(request.form['H-off_ops'])
            H_pit_whip = float(request.form['H-pit_whip'])
            H_pit_k_9 = float(request.form['H-pit_k/9'])
            V_off_avg = float(request.form['V-off_avg'])
            V_off_ops = float(request.form['V-off_ops'])
            V_pit_whip = float(request.form['V-pit_whip'])
            V_pit_k_9 = float(request.form['V-pit_k/9'])
        
        except:
            k = 0
            H_off_avg = float(0)
            H_off_ops = float(0)
            H_pit_whip = float(0)
            H_pit_k_9 = float(0)
            V_off_avg = float(0)
            V_off_ops = float(0)
            V_pit_whip = float(0)
            V_pit_k_9 = float(0)

        temp_array = temp_array + [H_off_avg, H_off_ops, H_pit_whip, H_pit_k_9, V_off_avg,
        V_off_ops, V_pit_whip, V_pit_k_9]

        data = np.array([temp_array])
        
        data2 = np.zeros((1,38),dtype=float)
        data2[:,:24]=data

        my_prediction = int(model.predict(data2))

        return render_template('home1.html', prediction_text=' {}'.format(my_prediction))

if __name__=="__main__":
    host = '0.0.0.0'
    port = 8080
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
    
