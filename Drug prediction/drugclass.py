# from flask import Flask,render_template,request
# import pickle
# import pandas as pd
# import numpy as np
# app = Flask(__name__)
# @app.route('/',methods = ['GET']) # route to display the home page
# def homepage():
#      return render_template('index.html')
# @app.route('/predict',methods=['POST'])# route to show the prediction in a web UI
# def index():
#     if request.method == 'POST':
#         try:
#             #reading the input given by the user
#             gender=request.form['gender']
#             bp = request.form['bp']
#             cholesterol = request.form['cholesterol']
#             age = request.form['age']
#             natok = request.form['natok']
#             inPut=np.array([[gender,bp,cholesterol,age,natok]])
#             print(inPut)
#             encoder = pickle.load(open('finalEncoder.pkl','rb'))
#             model = pickle.load(open('finalModel.pkl','rb'))
#             # prediction using the loaded model file
#             encoded_data = encoder.transform(inPut)
#             print(encoded_data)
#             prediction = model.predict(encoded_data)
#             print(prediction)
#             return render_template('result.html',prediction=prediction)
#         except Exception as e:
#             print('The excepting message is :',e)
#             return 'something is wrong'
#     else:
#         return render_template('index.html')
#
# if __name__=='__main__':
#     app.run(debug=True)#running the app

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)


@app.route('/', methods=['GET'])  # route to display the home page
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])  # route to show the prediction in a web UI
def index():
    if request.method == 'POST':
        try:
            # reading the input given by the user
            # gender = request.form['gender']
            # bp = request.form['bp']
            # cholesterol = request.form['cholesterol']
            # age = request.form['age']
            # natok = request.form['natok']
            # Inside the 'index' function
            gender = request.form['gender']
            bp = request.form['bp']
            cholesterol = request.form['cholesterol']
            age = request.form['age']
            natok = request.form['natok']
            print(f'Inputs: gender={gender}, bp={bp}, cholesterol={cholesterol}, age={age}, natok={natok}')

            inPut = np.array([[gender, bp, cholesterol, age, natok]])

            encoder = joblib.load(open('finalEncoder.pkl', 'rb'))
            model = joblib.load(open('finalModel.pkl', 'rb'))

            # prediction using the loaded model file
            encoded_data = encoder.transform(inPut)
            prediction = model.predict(encoded_data)

            return render_template('result.html', prediction=prediction[0])
        # except Exception as e:
        #     print('The exception message is:', e)
        #     return 'Something went wrong'
        except Exception as e:
            print(f'An error occurred: {e}')
            return 'Something went wrong'

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)  # running the app
