
import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle as p
import pickle
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from numpy import asarray
from keras_facenet import FaceNet

app = Flask(__name__)
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.embeddings(samples)
	return yhat[0]
#   Load Model 
modelfile = 'models/final_prediction.pickle'  
model1 = p.load(open(modelfile, 'rb'))
in_encoder= pickle.load(open('models/scaler.pickle','rb'))

@app.route('/')
def welcome():
    return render_template('index.html') 

@app.route('/upload',methods =['GET','POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']

    if image_file.filename == '':
        return redirect(request.url)

    if image_file:
        image_path = os.path.join('static/upload', image_file.filename)
        image_file.save(image_path)
        path = image_path
        face = extract_face(path)
        # Load model
        model = FaceNet()
        print('Loaded Model')
        # convert each face in the train set to an embedding
        face_pixels =face
        sample1 = get_embedding(model, face_pixels)
        sample1= expand_dims(sample1, axis=0)
        sample1 = in_encoder.transform(sample1)
        prediction = model1.predict((sample1))
        prediction=prediction[0]       
        # return render_template('index.html', image_path=image_path, predictions=prediction)
    if prediction==0:
        return render_template('index.html',image_path=image_path,predictions="ben_afflek")   
    if prediction==1:
        return render_template('index.html',image_path=image_path,predictions="elton_john")
    if prediction==2:
        return render_template('index.html',image_path=image_path,predictions="jerry_seinfeld") 
    if prediction==3:   
        return render_template('index.html',image_path=image_path,predictions="madonna")   
    else: 
        return render_template('index.html',image_path=image_path,predictions="mindy_kaling")
if __name__ == '__main__':
    app.run(debug=True)
