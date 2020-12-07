from flask import Flask
from flask import render_template, request
import os

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

if __name__=='__main__':
    app.run(port=1200,debug=True)
    