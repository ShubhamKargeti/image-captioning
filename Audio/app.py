from flask import Flask
from flask import render_template, request,send_from_directory
import os

from audio import *

# playsound("text.mp3")

app = Flask(__name__)
upload_folder = "/home/rahul/Self_ML_try/image_captioning/static"

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

@app.route("/", methods= ["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(upload_folder,image_file.filename)
            image_file.save(image_location)
    return render_template("index.html",prediction=1)

@app.route("/home/rahul/Self_ML_try/image_captioning/<path:filename>",methods=["GET"])
def download_file(filename):
    if request.method == "GET":
        return_speech(text="My Name is Rahul")
    return send_from_directory("/home/rahul/Self_ML_try/image_captioning/",filename)

if __name__=='__main__':
    app.run(port=1200,debug=True)
    