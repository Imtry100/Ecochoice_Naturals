# --- [Your imports and initializations: unchanged] ---
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
import torchvision.transforms as transforms
import joblib
from torchvision import models

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

MODEL_PATH = r"D:\Jhumpa Final\Website\fruit_model_1200each.pth"
ENCODER_PATH = r"D:\Jhumpa Final\Website\fruit_label_encoder.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("🔄 Loading model and label encoder...")
try:
    class_names = joblib.load(ENCODER_PATH)
    num_classes = len(class_names)
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_image(image_path):
    if model is None or not class_names:
        raise ValueError("Model or encoder not loaded properly")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, 1).item()
        predicted_label = class_names[predicted_idx]

    # Extract fruit name
    if predicted_label.startswith("fresh"):
        fruit = predicted_label.replace("fresh", "").strip("_")
    elif predicted_label.startswith("rotten"):
        fruit = predicted_label.replace("rotten", "").strip("_")
    else:
        fruit = predicted_label

    if fruit.lower() == "oranges":
        fruit = "orange"

    # Build mapping of fruit to fresh/rotten class indices
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
        raise ValueError(f"Unknown fruit type '{fruit}' from label '{predicted_label}'")

    # Now calculate freshness using logit difference
    fresh_idx = fruit_pairs[fruit]['fresh']
    rotten_idx = fruit_pairs[fruit]['rotten']
    logit_vals = logits[0].cpu().numpy()
    diff = logit_vals[fresh_idx] - logit_vals[rotten_idx]
    freshness_score = sigmoid(diff) * 100

    return predicted_label, freshness_score, fruit


# ------------------- ROUTES ----------------------

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

            corrections = {"patato": "potato", "apples": "apple", "oranges": "orange"}
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
        print("[DEBUG] Feedback payload received:", data)

        prediction = data.get('prediction')
        image_data = data.get('image_data')
        rating = data.get('freshness_score', '111')

        # Extra debugging
        if prediction is None:
            raise ValueError("Missing 'prediction'")
        if image_data is None:
            raise ValueError("Missing 'image_data'")

        prediction_folder = prediction.lower().replace(" ", "")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Save image
        folder_path = os.path.join(CORRECTION_FOLDER, prediction_folder)
        os.makedirs(folder_path, exist_ok=True)
        image_filename = f"{uuid.uuid4().hex[:12]}.jpg"
        image_path = os.path.join(folder_path, image_filename)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        print(f"[✓] Feedback image saved to: {image_path}")

        # Save to Excel
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
        print("[✓] Feedback saved to Excel.")
        return jsonify({"success": True})

    except Exception as e:
        print(f"[✗] Error in submit_feedback: {e}")
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

# --- START SERVER ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
