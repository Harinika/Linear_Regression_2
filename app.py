from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
# import pandas
app = Flask(__name__)
model = pickle.load(open('linear_regression_model1.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

standard_to = MinMaxScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # df=pd.read_csv('CarPrice_Assignment.csv')
        # df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
        # num_vars = ['price','carwidth','horsepower']
        # df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

        carwidth2 = float(request.form['carwidth'])
        # carwidth=standard_to.transform(carwidth2)

        horsepower2=float(request.form['horsepower'])
        # horsepower=standard_to.fit_transform(horsepower2)

        four=int(request.form['four'])
        highlevel=int(request.form['highlevel'])
        hatchback=int(request.form['hatchback'])
        dohcv=int(request.form['dohcv'])

        prediction=model.predict([[carwidth2,horsepower2,hatchback,dohcv,four,highlevel]])
        b= 45400
        a= 5118
        output=round(prediction[0]*(b-a)+a,2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

