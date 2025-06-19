import os
import uuid
import math
import base64
import re
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

UPLOAD_FOLDER = 'uploads'
CORRECTION_FOLDER = 'corrections'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CORRECTION_FOLDER, exist_ok=True)

PCA_MODEL_PATH = r"D:\\Data For Ecochoice Naturals\\FA\\pca_model.pkl"
ENCODER_PATH = r"D:\\Data For Ecochoice Naturals\\FA\\label_encoder.pkl"
SVM_MODEL_PATH = r"D:\\Data For Ecochoice Naturals\\FA\\svm_model.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading models...")
try:
    pca = joblib.load(PCA_MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    svm = joblib.load(SVM_MODEL_PATH)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    pca = encoder = svm = None

def get_feature_model(base_model, layer_name):
    model = base_model.eval().to(DEVICE)
    if layer_name == 'avgpool':
        return torch.nn.Sequential(*(list(model.children())[:-1]))
    elif layer_name == 'features':
        return model.features
    return model

print("Loading pretrained models...")
googlenet = get_feature_model(models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1), 'avgpool')
resnext = get_feature_model(models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1), 'avgpool')
densenet = get_feature_model(models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1), 'features')
print("Pretrained models loaded!")

def extract_features(model, inputs, model_type):
    with torch.no_grad():
        inputs = inputs.to(DEVICE)
        feats = model(inputs)
        if model_type == 'densenet':
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        return feats.view(inputs.size(0), -1).cpu()

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    if pca is None or encoder is None or svm is None:
        raise ValueError("Models not loaded properly")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    f1 = extract_features(googlenet, image_tensor, 'googlenet')
    f2 = extract_features(resnext, image_tensor, 'resnext')
    f3 = extract_features(densenet, image_tensor, 'densenet')
    feature_vector = torch.cat([f1, f2, f3], dim=1).numpy()

    reduced_vector = pca.transform(feature_vector)
    y_pred = svm.predict(reduced_vector)
    pred_label = encoder.inverse_transform(y_pred)[0]

    if pred_label.startswith("fresh"):
        fruit = pred_label.replace("fresh", "").strip("_")
    elif pred_label.startswith("rotten"):
        fruit = pred_label.replace("rotten", "").strip("_")
    else:
        fruit = pred_label

    if fruit.lower() == "oranges":
        fruit = "orange"

    class_names = encoder.classes_
    fruit_pairs = {}
    for i, name in enumerate(class_names):
        if name.startswith("fresh"):
            fname = name.replace("fresh", "").strip("_")
            if fname == "oranges": fname = "orange"
            fruit_pairs.setdefault(fname, {})['fresh'] = i
        elif name.startswith("rotten"):
            fname = name.replace("rotten", "").strip("_")
            if fname == "oranges": fname = "orange"
            fruit_pairs.setdefault(fname, {})['rotten'] = i

    if fruit not in fruit_pairs:
        raise ValueError(f"Unknown fruit type '{fruit}' from label '{pred_label}'")

    decision_scores = svm.decision_function(reduced_vector)
    fresh_idx = fruit_pairs[fruit]['fresh']
    rotten_idx = fruit_pairs[fruit]['rotten']
    diff = decision_scores[0][fresh_idx] - decision_scores[0][rotten_idx]
    freshness_score = sigmoid(diff) * 100

    return pred_label, freshness_score, fruit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            pred_label, freshness_score, fruit = predict_image(filepath)
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            os.remove(filepath)

            if freshness_score >= 70:
                status = "Fresh"
                status_color = "success"
            elif freshness_score >= 40:
                status = "Moderately Fresh"
                status_color = "warning"
            else:
                status = "Rotten"
                status_color = "danger"

            corrections = {"patato": "potato",
                           "apples" : "apple",
                           "oranges": "orange"}
            fruit_cleaned = fruit.lower()
            fruit_corrected = corrections.get(fruit_cleaned, fruit_cleaned).title()

            if pred_label.lower().startswith("fresh"):
                formatted_prediction = "Fresh " + fruit_corrected
            elif pred_label.lower().startswith("rotten"):
                formatted_prediction = "Rotten " + fruit_corrected
            else:
                formatted_prediction = fruit_corrected

            return render_template('result.html',
                                   prediction=formatted_prediction,
                                   freshness_score=round(freshness_score, 2),
                                   fruit=fruit_corrected,
                                   status=status,
                                   status_color=status_color,
                                   image_data=img_base64)

        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(url_for('index'))

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        print("[INFO] Feedback Data:", data)

        prediction = data.get('prediction')
        image_data = data.get('image_data')
        rating = data.get('freshness_score', '111')

        if not prediction or not image_data:
            raise ValueError("Missing 'prediction' or 'image_data'")

        prediction_folder = prediction.lower().replace(" ", "")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        folder_path = os.path.join(CORRECTION_FOLDER, prediction_folder)
        os.makedirs(folder_path, exist_ok=True)
        image_filename = f"{uuid.uuid4().hex[:12]}.jpg"
        image_path = os.path.join(folder_path, image_filename)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        print(f"[✓] Feedback image saved to: {image_path}")

        excel_path = os.path.join(CORRECTION_FOLDER, 'corrections.xlsx')
        fruit = re.sub(r'^(fresh|rotten)', '', prediction_folder).title()
        df_new = pd.DataFrame([{
            'Image Name': image_filename,
            'Condition': prediction_folder,
            'Fruit': fruit,
            'Freshness Score (0-100)': rating,
            'User Said': 'Correct'
        }])
        if os.path.exists(excel_path):
            df_existing = pd.read_excel(excel_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_excel(excel_path, index=False)
        return jsonify({"success": True})

    except Exception as e:
        print(f"[✗] Error in feedback save: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/submit_correction', methods=['POST'])
def submit_correction():
    try:
        data = request.get_json()
        actual_condition = data['actual_condition'].lower()
        actual_item = data['actual_item'].lower()
        rating = data['rating']
        image_data = data['image_data']

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        folder_name = f"{actual_condition}{actual_item}"
        folder_path = os.path.join(CORRECTION_FOLDER, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        image_filename = f"{uuid.uuid4().hex[:12]}.jpg"
        image_path = os.path.join(folder_path, image_filename)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        excel_path = os.path.join(CORRECTION_FOLDER, 'corrections.xlsx')
        df_new = pd.DataFrame([{
            'Image Name': image_filename,
            'Condition': folder_name,
            'Fruit': actual_item.title(),
            'Freshness Score (0-100)': rating,
            'User Said': 'Incorrect'
        }])
        if os.path.exists(excel_path):
            df_existing = pd.read_excel(excel_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_excel(excel_path, index=False)
        return jsonify({"success": True})

    except Exception as e:
        print(f"[✗] Error in correction save: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            pred_label, freshness_score, fruit = predict_image(filepath)
            os.remove(filepath)

            if freshness_score >= 70:
                status = "Fresh"
            elif freshness_score >= 40:
                status = "Moderately Fresh"
            else:
                status = "Rotten"

            return jsonify({
                'prediction': pred_label,
                'freshness_score': round(freshness_score, 2),
                'fruit': fruit.title(),
                'status': status,
                'success': True
            })

        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 500

    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
