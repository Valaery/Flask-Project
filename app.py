from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Ruta al modelo TFLite
MODEL_PATH = 'model_classifier_image.tflite'

# Cargar el modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

def preprocess_image(image_path):
    # Cargar y preprocesar la imagen
    img = Image.open(image_path).convert('RGB')
    img = img.resize((100, 100))  # Cambiar el tamaño a 100x100
    img_array = np.array(img) / 255.0  # Normaliza la imagen
    return img_array

def model_predict(file_path):
    image = preprocess_image(file_path)
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    # Configurar el intérprete para la inferencia
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Obtener la predicción
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(upload_path)

        prediction = model_predict(upload_path)
        # Asumir que prediction es un array con las probabilidades
        result = 'Dog' if prediction[0] > 0.5 else 'Cat'

        # Generar HTML para mostrar la imagen y el resultado
        img_tag = f'<img src="/uploads/{secure_filename(f.filename)}" style="max-width: 300px;"><br>'
        result_html = f'<p>Prediction: {result}</p><br>'
        retry_button = '<button onclick="window.location.reload();">Try Another Image</button>'
        return img_tag + result_html + retry_button

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)