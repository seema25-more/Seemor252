from flask import Flask, request, jsonify, render_template
import pickle
import keras
from tensorflow import keras
import tensorflow as tf
import numpy as np

# Create flask app
app = Flask(__name__,template_folder='template')

# Load the pickle model
model = pickle.load(open(r"C:\Seema project\model_3.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html")
    app.run(debug=True)


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text="Prediction Class is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)