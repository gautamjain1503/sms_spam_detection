from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from sms_spam_detection.pipeline.stage_3_predict import Predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret_base_key"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def validator():
    if request.method=='POST':
        message = request.form['Message']
        obj = Predict()
        result=obj.main(message=message)
        return render_template('index.html', result=result, message=message)

    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host= '0.0.0.0', debug=False)