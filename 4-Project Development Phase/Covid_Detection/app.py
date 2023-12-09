"""
import re
import numpy as np
import os
from flask import Flask, app,request,render_template
from tensorflow import keras
from keras import models
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for


#Loading the model

model = load_model(r"covid.h5")
app=Flask(__name__)

#Render HTML pages
@app.route('/')
def index():
    return render_template('Test.html')


@app.route('/result')
def  home():
    return render_template("Test.html")


@app.route("/result", methods=["GET","POST"])
def res():
    if request.method == "POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)

        filepath=os.path.join(basepath,'upload',f.filename)

        f.save(filepath)

        img=image.load_img(filepath,target_size=(500,500))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)

        img_data=preprocess_input(x)
        prediction=np.argmax(model.predict(img_data), axis=1)

        index=['COVID','NON-COVID']

        result=str(index[prediction[0]])
        print(result)
        return render_template('Test.html',prediction=result)
    
if __name__=='__main__':
    app.run(debug = False)

"""
import os
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('covid.h5')

# Define the classes for binary classification (COVID-19 or non-COVID-19)
class_names = ['Non-COVID-19', 'COVID-19']

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/About')
def about():
    return render_template('About.html')

@app.route('/Precautions')
def precautions():
    return render_template('Precautions.html')

@app.route('/Vaccinations')
def vaccinations():
    return render_template('Vaccinations.html')

@app.route('/Test')
def test():
    return render_template('Test.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('Test.html', prediction_text='No file part')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('Test.html', prediction_text='No selected file')

    if file and allowed_file(file.filename):
        # Load and preprocess the image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)
        img = image.load_img(img_path, target_size=(500, 500))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array[:, :, :, 0], axis=-1)
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)
        prediction = class_names[int(predictions[0, 0] > 0.5)]

        # Display the result on the web page
        return render_template('Test.html', prediction_text=f'The image is classified as: {prediction}')

    else:
        return render_template('Test.html', prediction_text='Invalid file format')

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', debug=True)
