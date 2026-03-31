from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([data])
        return f"Prediction: {prediction[0]}"
    return render_template("index.html")

app.run(debug=True)
