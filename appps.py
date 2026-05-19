from flask import Flask, render_template, request
import numpy as np
import os
import wfdb
from keras.models import load_model

app = Flask(__name__)

model = load_model("modell.keras", compile=False)

classes = ['/', 'A', 'E', 'L', 'N', 'R', 'V']

class_names = {
    '/': 'Paced Beat',
    'A': 'Atrial Premature Beat',
    'E': 'Ventricular Escape Beat',
    'L': 'Left Bundle Branch Block Beat',
    'N': 'Normal Beat',
    'R': 'Right Bundle Branch Block Beat',
    'V': 'Ventricular Premature Beat'
}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    dat_file = request.files["dat_file"]
    hea_file = request.files["hea_file"]
    start_index = int(request.form["start_index"])

    dat_path = os.path.join(UPLOAD_FOLDER, dat_file.filename)
    hea_path = os.path.join(UPLOAD_FOLDER, hea_file.filename)

    dat_file.save(dat_path)
    hea_file.save(hea_path)

    # record name without extension
    record_name = hea_file.filename.replace(".hea", "")
    record_path = os.path.join(UPLOAD_FOLDER, record_name)

    # read ECG record
    record = wfdb.rdrecord(record_path)

    # take first ECG channel
    signal = record.p_signal[:, 0]

    # take 360 values from start index
    segment = signal[start_index:start_index + 360]

    if len(segment) != 360:
        return render_template(
            "index.html",
            result="Invalid start index. Not enough ECG values after this index."
        )

    input_data = np.array(segment).reshape(1, 360, 1)

    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]

    result = class_names[predicted_class]

    if predicted_class == 'N':
        status = "Heartbeat appears normal."
    else:
        status = "Yes, arrhythmia is detected."

    return render_template(
        "index.html",
        result=result,
        symbol=predicted_class,
        status=status
    )

if __name__ == "__main__":
    app.run(debug=True)