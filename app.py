from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.cm as cm

app = Flask(__name__)

# -------------------------
# Load Models
# -------------------------

chest_model = tf.keras.models.load_model("radiology_model.h5")
fracture_model = tf.keras.models.load_model("radiology_model2.h5")
ultrasound_model = tf.keras.models.load_model("radiology_model3.h5")

# -------------------------
# Class Labels
# -------------------------

chest_classes = ["COVID", "Normal", "Viral Pneumonia"]
fracture_classes = ["fracture", "normal"]
ultrasound_classes = ["benign", "malignant", "normal"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------
# Image Preprocessing
# -------------------------

def preprocess(img_path):

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# -------------------------
# GradCAM Heatmap
# -------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# -------------------------
# Save Heatmap Image
# -------------------------

def save_heatmap(img_path, heatmap):

    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    heatmap_path = img_path.replace(".png","_heatmap.png").replace(".jpg","_heatmap.jpg")

    cv2.imwrite(heatmap_path,superimposed)

    return heatmap_path


# -------------------------
# Page 1 : Patient Details
# -------------------------

@app.route("/")
def home():
    return render_template("patient.html")


# -------------------------
# Page 2 : Scan Type
# -------------------------

@app.route("/select_scan", methods=["POST"])
def select_scan():

    name = request.form["name"]
    age = request.form["age"]
    gender = request.form["gender"]

    return render_template(
        "scan_type.html",
        name=name,
        age=age,
        gender=gender
    )


# -------------------------
# Page 3 : Upload Image
# -------------------------

@app.route("/upload/<scan_type>")
def upload(scan_type):

    return render_template(
        "upload.html",
        scan_type=scan_type
    )


# -------------------------
# Page 4 : Prediction
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():

    scan_type = request.form["scan_type"]
    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = preprocess(path)

    # -------------------------
    # Chest Model
    # -------------------------

    if scan_type == "chest":

        pred = chest_model.predict(img)
        label = chest_classes[np.argmax(pred)]

        heatmap = make_gradcam_heatmap(img, chest_model, "Conv_1")

    # -------------------------
    # Fracture Model
    # -------------------------

    elif scan_type == "fracture":

        pred = fracture_model.predict(img)
        label = fracture_classes[np.argmax(pred)]

        heatmap = make_gradcam_heatmap(img, fracture_model, "Conv_1")

    # -------------------------
    # Ultrasound Model
    # -------------------------

    elif scan_type == "ultrasound":

        pred = ultrasound_model.predict(img)
        label = ultrasound_classes[np.argmax(pred)]

        heatmap = make_gradcam_heatmap(img, ultrasound_model, "Conv_1")

    # Save heatmap image
    heatmap_path = save_heatmap(path, heatmap)

    return render_template(
        "result.html",
        prediction=label,
        image=path,
        heatmap=heatmap_path,
        scan_type=scan_type
    )


# -------------------------

if __name__ == "__main__":
    app.run(debug=True)