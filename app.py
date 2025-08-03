from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = tf.keras.models.load_model('asl_model.h5')
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'DEL', 'NOTHING', 'SPACE']
IMG_SIZE = 64
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        print(f"Received file: {filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = Image.open(filepath).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, IMG_SIZE, IMG_SIZE, 3))

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return render_template('result.html', filename=filename,
                               prediction=predicted_class, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
