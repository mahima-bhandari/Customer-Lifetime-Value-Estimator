from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)
loaded_model = joblib.load('linear_modelled.pk1')

@app.route('/')
def home():
    return render_template('clve_interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    avg_session_length = float(request.form['avgSessionLength'])
    time_on_app = float(request.form['timeOnApp'])
    time_on_website = float(request.form['timeOnWebsite'])
    length_of_membership = float(request.form['lengthOfMembership'])

    input_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
    predicted_yearly_amount_spent = loaded_model.predict(input_data)

    return redirect(url_for('result', prediction=predicted_yearly_amount_spent[0]))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=8081)  