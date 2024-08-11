from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
application = Flask(__name__)
app = application

# Load the model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scalr = pickle.load(open("models/scaler.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        # Extracting form data
        try:
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            DC = float(request.form['DC'])
            ISI = float(request.form['ISI'])
            BUI = float(request.form['BUI'])
            FWI = float(request.form['FWI'])
            Classes = int(request.form['Classes'])
            Region = int(request.form['Region'])  # Added this line

            # Combine inputs into a single array for prediction, now including 'Region'
            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI, Classes, Region]])

            # Scale the input data
            scaled_data = standard_scalr.transform(input_data)

            # Make prediction using the model
            result = ridge_model.predict(scaled_data)

            # Render the prediction result on the home.html page
            return render_template('home.html', result=result[0])
        
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
