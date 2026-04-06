import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import base64
from io import BytesIO, StringIO
# Tumor
import imutils
import matplotlib as mpl
from PIL import Image
import keras


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'model'
app.config['PNEUMONIA_FOLDER'] = 'Pneumonia'

##heatmap
def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return cam_path
##

#-----------------Crop
def crop_img(img, add_pixels_value=0):
    """
    Crops the image around its largest contour.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use BGR to GRAY since OpenCV loads BGR
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold, erode, dilate
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours and the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return img  # fallback if no contours found

    c = max(cnts, key=cv2.contourArea)

    # Extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = add_pixels_value
    new_img = img[max(extTop[1]-ADD_PIXELS, 0):extBot[1]+ADD_PIXELS,
                  max(extLeft[0]-ADD_PIXELS, 0):extRight[0]+ADD_PIXELS].copy()
    return new_img



# ---------- ROUTES ----------
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route("/pneumonia", methods=["GET", "POST"])
def pneumonia():
    models = [f for f in os.listdir('pneumonia') if f.endswith('.tflite')]
    IMG_SIZE = (224, 224)

    if request.method == "POST":
        selected_model = request.form.get("model_name")
        model_path = os.path.join('pneumonia', selected_model)

        file = request.files.get("image")
        if not file:
            return render_template("pneumonia.html", models=models, selected_model=selected_model)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (224, 224))
        input_data = np.expand_dims(resized.astype(np.float32), axis=0)

        # Load model and predict
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        label = "PNEUMONIA" if prediction>= 0.98 else "NORMAL"
        print(prediction)
        confidence = round(float(prediction), 3)

        #heatMap
        model_path_hm = os.path.join(app.config['PNEUMONIA_FOLDER'],'ResNet50_Pneumonia_FI.h5' )
        model_hm = keras.models.load_model(model_path_hm)
        model_hm.layers[-1].activation = None  # Needed for Grad-CAM
        img_array = get_img_array(filepath, IMG_SIZE) / 255.0
  
        heatmap = make_gradcam_heatmap(img_array, model_hm, last_conv_layer_name="conv5_block3_out")  # change layer name as needed
        cam_path = os.path.join(app.config['UPLOAD_FOLDER'], "heatmap.jpg")
        cam_file = save_and_display_gradcam(filepath, heatmap, cam_path=cam_path)

        cam_image = cv2.imread(cam_path)
        _, buffer = cv2.imencode('.jpg', cam_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template("pneumonia.html",
                               label=label,
                               confidence=confidence,
                               img_data=img_base64,
                               models=models,
                               selected_model=selected_model)

    return render_template("pneumonia.html",
                           models=models,
                           label=None,
                           img_data=None,
                           selected_model=None)



####

@app.route("/download", methods=["POST"])
def download():
    global last_csv
    if last_csv:
        return send_file(BytesIO(last_csv.getvalue().encode()), mimetype='text/csv',
                         as_attachment=True, download_name='diagnosis_results.csv')
    return "No data", 400

if __name__ == "__main__":
    last_csv = None
    app.run(debug=True)
