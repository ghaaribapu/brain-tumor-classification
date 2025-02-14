import os
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import imutils
import google.generativeai as genai
from sklearn.decomposition import PCA
import plotly.express as px

# =============================================================================
# Global Constants and Settings
# =============================================================================
st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/brain-tumor-detection',
        'Report a bug': "https://github.com/yourusername/brain-tumor-detection/issues",
        'About': "# Brain Tumor Detection System v1.0\nDeveloped by the APU Team"
    }
)

MODEL_ACCURACIES = {
    "tf_efficientnet_model": 90.10,
    "tf_vgg16_model.h5": 88.72,
    "tf_inception_model.h5": 82.29,
    "tf_resnet50_model.h5": 84.90,
    "ga_mlp_model.h5": 62.33,
    "pytorch_resnet50_model.pth": 81.25,
    "rf_model.pkl": 83.33,
    "svm_model.pkl": 86.81,
    "lr_model.pkl": 71.01
}

CLASS_NAMES = ["glioma", "notumor", "meningioma", "pituitary"]

DATASET_DIR = "brain_tumor_mris"
TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
VAL_DIR = os.path.join(DATASET_DIR, "Validation")
TEST_DIR = os.path.join(DATASET_DIR, "Testing")

# Configure Gemini API – ensure your API key is valid.
GEMINI_API_KEY = "AIzaSyCyyvz5lm5uitFaPmaY0pBqulf2Lb1oBrI"
genai.configure(api_key=GEMINI_API_KEY)

# =============================================================================
# Helper Functions for Preprocessing, Visualization & Prediction
# =============================================================================
def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    image = img_as_ubyte(image)
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.append(graycoprops(glcm, prop).mean())
    features.append(np.var(image))
    features.append(np.mean(image))
    features.append(np.std(image))
    return np.array(features)

def extract_glcm_features_from_image(pil_image):
    img = pil_image.convert("L")
    return extract_glcm_features(np.array(img))

def apply_red_overlay(image):
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    enhancer = ImageEnhance.Contrast(Image.fromarray(img_np))
    img_enhanced = np.array(enhancer.enhance(1.5))
    red_overlay = np.full(img_np.shape, (255, 0, 0), dtype=np.uint8)
    red_img = cv2.addWeighted(img_enhanced, 0.7, red_overlay, 0.3, 0)
    return red_img

def pytorch_transform(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform_pipeline(image)

# ---------------- Advanced Image Analysis Functions ----------------
# These functions operate solely on the uploaded image.
def visualize_extreme_points(image, add_pixels_value=0):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        gray = img_np
    elif img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unexpected image format")
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return img_np, img_np
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    extTop = tuple(c[c[:,:,1].argmin()][0])
    extBot = tuple(c[c[:,:,1].argmax()][0])
    img_contour = img_np.copy()
    cv2.drawContours(img_contour, [c], -1, (0, 255, 255), 2)
    cv2.circle(img_contour, extLeft, 5, (255, 0, 0), -1)
    cv2.circle(img_contour, extRight, 5, (0, 255, 0), -1)
    cv2.circle(img_contour, extTop, 5, (0, 0, 255), -1)
    cv2.circle(img_contour, extBot, 5, (255, 255, 0), -1)
    cropped_img = img_np[extTop[1]-add_pixels_value:extBot[1]+add_pixels_value,
                          extLeft[0]-add_pixels_value:extRight[0]+add_pixels_value]
    return img_contour, cropped_img

def detect_tumor_bounding_box(img, threshold=200):
    if len(img.shape) == 2:
        gray = img
    elif img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unexpected image format")
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return img, None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    img_box = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_box, (x, y, w, h)

def visualize_tumor(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        gray = img_np
    elif img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unexpected image format")
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    image_area = img_np.shape[0] * img_np.shape[1]
    valid_contours = [cnt for cnt in cnts if cv2.contourArea(cnt) < 0.5 * image_area]
    if valid_contours:
        c = max(valid_contours, key=cv2.contourArea)
    elif cnts:
        c = max(cnts, key=cv2.contourArea)
    else:
        return img_np, img_np, img_np, None
    x, y, w, h = cv2.boundingRect(c)
    cropped_img = img_np[y:y+h, x:x+w]
    tumor_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    bounding_box_img = img_np.copy()
    cv2.rectangle(bounding_box_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(gray.ravel(), bins=50, color='steelblue', alpha=0.7)
    ax.set_title("Pixel Intensity Histogram")
    return cropped_img, tumor_mask, bounding_box_img, fig

# Advanced Analysis: Computes and displays a feature map plus other visuals.
def preprocess_and_visualize(image):
    """
    Enhanced preprocessing that:
      1. Computes and displays a feature map for the uploaded image.
      2. Extracts and displays extreme points.
      3. Applies CLAHE and Gaussian blur, then uses Otsu’s thresholding to generate a red tumor mask.
      4. Detects and overlays the tumor region bounding box.
      5. Displays a pixel intensity histogram.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image)
    
    # Compute extreme points.
    contour_img, _ = visualize_extreme_points(image)
    
    # Compute feature map using VGG16's block1_conv1.
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    layer_output = base_model.get_layer("block1_conv1").output
    truncated_model = tf.keras.Model(inputs=base_model.input, outputs=layer_output)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    feature_maps = truncated_model.predict(img_batch)
    feature_map_display = feature_maps[0, :, :, 0]
    
    # CLAHE, blur, and Otsu's thresholding.
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    red_mask[mask == 255] = [255, 0, 0]
    
    # Tumor bounding box.
    img_box, _ = detect_tumor_bounding_box(img_np)
    
    # Pixel intensity histogram.
    red_channel = img_np[..., 0]
    
    # Display in a 2x3 grid (enlarged).
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(contour_img)
    axes[0, 1].set_title("Extreme Points")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(feature_map_display, cmap="viridis")
    axes[0, 2].set_title("Feature Map (Channel 0)")
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(red_mask)
    axes[1, 0].set_title("Otsu's Thresholding (Red Mask)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(img_box)
    axes[1, 1].set_title("Tumor Bounding Box")
    axes[1, 1].axis("off")
    
    axes[1, 2].hist(red_channel.ravel(), bins=256, color='red', alpha=0.7)
    axes[1, 2].set_title("Pixel Intensity Histogram")
    axes[1, 2].set_xlabel("Intensity")
    axes[1, 2].set_ylabel("Count")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return feature_map_display

# ---------------- Model Loading and Prediction Functions ----------------
LOADED_MODELS = {}

def load_model_file(file_name):
    if file_name in LOADED_MODELS:
        return LOADED_MODELS[file_name]
    
    model_path = os.path.join("saved_models", file_name)
    if file_name.endswith(".pth"):
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        device = torch.device("cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        model_dict = {"type": "pytorch", "model": model, "device": device}
    elif file_name.endswith(".pkl"):
        package = joblib.load(model_path)
        model_dict = {"type": "sklearn", "model": package["model"],
                      "scaler": package.get("scaler"),
                      "poly": package.get("poly")}
    elif file_name == "ga_mlp_model.h5":
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model_dict = {"type": "tensorflow", "model": model}
    elif file_name.endswith(".h5") or file_name == "tf_efficientnet_model":
        if file_name.endswith(".h5"):
            custom_objects = {"TFOpLambda": tf.keras.layers.Lambda, "Functional": tf.keras.models.Model}
            model = tf_load_model(model_path, compile=False, custom_objects=custom_objects)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model_dict = {"type": "tensorflow", "model": model}
        else:
            json_path = os.path.join("saved_models", "tf_efficientnet_model.json")
            weights_path = os.path.join("saved_models", "tf_efficientnet_model_weights.h5")
            with open(json_path, "r") as json_file:
                model_json = json_file.read()
            custom_objects = {"InputLayer": tf.keras.layers.InputLayer, "Functional": tf.keras.models.Model}
            model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
            model.load_weights(weights_path)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model_dict = {"type": "tensorflow", "model": model}
    else:
        st.error("Unsupported model file format.")
        return None

    LOADED_MODELS[file_name] = model_dict
    return model_dict

def predict_with_model(model_dict, image):
    model_type = model_dict["type"]
    if model_type == "pytorch":
        device = model_dict["device"]
        model = model_dict["model"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        input_tensor = pytorch_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
        return int(pred.item())
    elif model_type == "tensorflow":
        model = model_dict["model"]
        if st.session_state.get("selected_model_file", "") == "ga_mlp_model.h5":
            features = extract_glcm_features_from_image(image).reshape(1, -1)
            preds = model.predict(features)
            return int(np.argmax(preds[0]))
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            preds = model.predict(img_batch)
            return int(np.argmax(preds[0]))
    elif model_type == "sklearn":
        model = model_dict["model"]
        scaler = model_dict.get("scaler")
        poly = model_dict.get("poly")
        features = extract_glcm_features_from_image(image).reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)
        if poly is not None:
            features = poly.transform(features)
        return int(model.predict(features)[0])
    else:
        st.error("Unknown model type.")
        return None

def get_gemini_recommendation(tumor_type):
    prompt = f"""
    Provide a detailed but easy-to-understand explanation about the brain tumor type: {tumor_type}.
    Explain:
      - What this tumor is
      - Potential causes
      - Common symptoms
      - Recommended medical steps for diagnosis and treatment
      - Advice for patients and caregivers
    Keep the answer concise, relevant, and non-alarming.
    """
    try:
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error fetching recommendation: {str(e)}"

def visualize_feature_maps(image):
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    layer_output = base_model.get_layer("block1_conv1").output
    truncated_model = tf.keras.Model(inputs=base_model.input, outputs=layer_output)
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    feature_maps = truncated_model.predict(img_batch)
    num_features = min(feature_maps.shape[-1], 16)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for i in range(16):
        if i < num_features:
            axes[i].imshow(feature_maps[0, :, :, i], cmap="viridis")
            axes[i].axis("off")
        else:
            axes[i].axis("off")
    st.pyplot(fig)

def pca_analysis():
    sample_features = []
    sample_labels = []
    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(TEST_DIR, cls)
        if os.path.isdir(cls_dir):
            image_files = os.listdir(cls_dir)[:10]
            for img_file in image_files:
                img_path = os.path.join(cls_dir, img_file)
                try:
                    img = Image.open(img_path)
                    features = extract_glcm_features_from_image(img)
                    sample_features.append(features)
                    sample_labels.append(cls)
                except Exception as e:
                    continue
    if len(sample_features) > 0:
        X = np.array(sample_features)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df_pca = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Class": sample_labels
        })
        fig = px.scatter(df_pca, x="PC1", y="PC2", color="Class",
                         title="PCA Analysis of GLCM Features")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No images found for PCA analysis.")

# =============================================================================
# Sidebar Model Selection (Grouped and Sorted)
# =============================================================================
def get_grouped_model_options():
    pt_models = [("pytorch_resnet50_model.pth", MODEL_ACCURACIES["pytorch_resnet50_model.pth"])]
    ga_models = [("ga_mlp_model.h5", MODEL_ACCURACIES["ga_mlp_model.h5"])]
    skl_models = [("svm_model.pkl", MODEL_ACCURACIES["svm_model.pkl"]),
                  ("rf_model.pkl", MODEL_ACCURACIES["rf_model.pkl"]),
                  ("lr_model.pkl", MODEL_ACCURACIES["lr_model.pkl"])]
    tf_models = [("tf_efficientnet_model", MODEL_ACCURACIES["tf_efficientnet_model"]),
                 ("tf_vgg16_model.h5", MODEL_ACCURACIES["tf_vgg16_model.h5"]),
                 ("tf_inception_model.h5", MODEL_ACCURACIES["tf_inception_model.h5"]),
                 ("tf_resnet50_model.h5", MODEL_ACCURACIES["tf_resnet50_model.h5"])]
    grouped = {
        "PyTorch": pt_models,
        "GA Models": ga_models,
        "Sklearn": skl_models,
        "TensorFlow": tf_models
    }
    return grouped

def sidebar_model_selector():
    grouped = get_grouped_model_options()
    selected = None
    st.sidebar.markdown("### 📌 Select Model")
    for group, models_list in grouped.items():
        st.sidebar.markdown(f"#### {group}")
        for model_name, acc in models_list:
            if st.sidebar.button(f"{model_name} ({acc:.1f}%)", key=f"{group}_{model_name}"):
                selected = model_name
    if selected is None:
        if len(get_grouped_model_options()["PyTorch"]) > 0:
            selected = get_grouped_model_options()["PyTorch"][0][0]
        else:
            selected = get_grouped_model_options()["TensorFlow"][0][0]
    st.session_state.selected_model_file = selected

# =============================================================================
# Sidebar Navigation and Home Page
# =============================================================================
def sidebar_nav():
    nav_option = st.sidebar.radio("", ["🏠 Home", "🔍 Predict", "🖼 Advanced Analysis", "💬 AI Chat", "📊 Evaluation", "🎨 Feature Maps"], index=0)
    st.sidebar.markdown("---")
    sidebar_model_selector()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 Model Performance")
    for model, acc in MODEL_ACCURACIES.items():
        st.sidebar.progress(acc / 100, f"{model}: {acc:.1f}%")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👥 Team")
    st.sidebar.markdown("""
    - 👨‍💻 Ghaarib Khurshid  
    - 👨‍💻 Ahmed Ashfaq  
    - 👨‍🏫 Dr. Nirasee Fathima
    """)
    return nav_option

def home_page():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🧠 Brain Tumor Detection System")
        st.markdown("""
        <div style='background-color: #112936; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #ff4b4b;'>🎯 Our Mission</h2>
            <p>Leveraging cutting-edge AI technology to provide accurate and rapid brain tumor detection through MRI scan analysis using multiple advanced models:
            <br><strong>EfficientNetB3, GA‑tuned VGG16, InceptionV3, ResNet50, GA‑optimized MLP, RandomForest, SVM, Logistic Regression,</strong> and <strong>PyTorch ResNet50</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🚀 Key Features")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            #### 🤖 AI Models
            - EfficientNetB3 (TF)
            - GA‑Tuned VGG16 (TF)
            - InceptionV3 (TF)
            - ResNet50 (TF)
            - PyTorch ResNet50
            - GA‑Optimized MLP
            - RandomForest
            - SVM
            - Logistic Regression
            """)
        with col_b:
            st.markdown("""
            #### 📊 Analysis
            - GLCM Feature Extraction
            - Extreme Points Detection
            - Otsu's Thresholding
            - PCA Analysis on GLCM Features
            - Feature Maps Visualization
            """)
        with col_c:
            st.markdown("""
            #### 💡 Intelligence
            - AI Recommendations via Gemini API
            - Multi-model Prediction
            - Comprehensive Performance Metrics
            """)
    with col2:
        st.image("tumor.jpg", use_container_width=True)

# ---------------- Prediction Page ----------------
def predict_page():
    st.title("🔍 Tumor Detection Analysis")
    prediction_container = st.container()
    col1, col2 = st.columns(2)
    with col1, st.expander("📝 Instructions", expanded=True):
        st.markdown("Upload an MRI scan (a large preview will be shown).")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_image = original_image
        preview_image = original_image.copy()
        preview_image.thumbnail((600, 600))
        with st.spinner("🔄 Analyzing image..."):
            model_dict = load_model_file(st.session_state.selected_model_file)
            if model_dict is not None:
                pred_idx = predict_with_model(model_dict, original_image)
                result_class = CLASS_NAMES[pred_idx]
                st.session_state.result_class = result_class
            else:
                result_class = "notumor"
        color = "#ff4b4b" if result_class.lower() != "notumor" else "#00cc66"
        with prediction_container:
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h3>Prediction Result</h3>
                <h2>{result_class.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col1:
            st.image(preview_image, caption="📸 Uploaded MRI Scan (Large Preview)", use_container_width=True)

# ---------------- Advanced Analysis Page ----------------
def advanced_analysis_page():
    st.title("🖼 Advanced Analysis")
    st.markdown("This page displays advanced image analysis computed from the uploaded image.")
    if "original_image" not in st.session_state or st.session_state.original_image is None:
        st.warning("No image found. Please upload an image on the Predict page first.")
        return
    original_image = st.session_state.original_image
    with st.spinner("Generating advanced analysis..."):
        preprocess_and_visualize(original_image)

# ---------------- AI Chat Page ----------------
def ai_chat_page():
    st.title("💬 AI Chat")
    st.markdown("This page provides a Gemini AI recommendation based on the predicted tumor type from your uploaded image.")
    if "result_class" not in st.session_state or st.session_state.result_class is None:
        st.warning("No prediction found. Please upload an image on the Predict page first.")
        return
    tumor_type = st.session_state.result_class
    st.markdown(f"**Predicted Tumor Type:** {tumor_type}")
    if st.button("Get AI Recommendation"):
        with st.spinner("Fetching AI recommendation..."):
            recommendation = get_gemini_recommendation(tumor_type)
        chat_html = f"""
        <div style="margin: 10px 0; display: flex; flex-direction: column;">
          <div style="align-self: flex-start; background-color: #e1ffc7; color: #000; padding: 15px; border-radius: 10px; max-width: 80%;">
             <strong>Assistant:</strong> {recommendation}
          </div>
        </div>
        """
        st.markdown(chat_html, unsafe_allow_html=True)

# ---------------- Evaluation Page ----------------
def evaluation_page():
    st.title("📊 Model Performance Analysis")
    df_perf = pd.DataFrame({
        "Model": list(MODEL_ACCURACIES.keys()),
        "Accuracy": list(MODEL_ACCURACIES.values())
    })
    df_perf = df_perf.sort_values("Accuracy", ascending=False)
    fig = px.bar(
        df_perf, x="Model", y="Accuracy", color="Accuracy", text="Accuracy",
        title="Model Performance Comparison", color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🎯 Detailed Metrics")
    baseline = 80.0
    model_data = []
    for model, acc in MODEL_ACCURACIES.items():
        delta_val = acc - baseline
        arrow = "↑" if delta_val >= 0 else "↓"
        delta_text = f"{arrow} {abs(delta_val):.1f}%" if delta_val >= 0 else f"{arrow} {abs(delta_val):.1f}% below baseline"
        model_data.append((model, acc, delta_text))
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    for row in chunk_list(model_data, 3):
        cols = st.columns(len(row))
        for col, (label, val, delta) in zip(cols, row):
            color = "#00cc66" if "↑" in delta else "#ff4b4b"
            col.markdown(
                f"""
                <div style="background-color: #112936; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
                    <h4 style="color: #ffffff; margin: 0;">{label}</h4>
                    <h2 style="color: {color}; margin: 8px 0 4px 0;">{val:.1f}%</h2>
                    <p style="color: #ffffff; margin: 0;">{delta}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("---")
    st.markdown("### 📈 PCA Analysis on GLCM Features")
    if st.checkbox("Show PCA Analysis", key="pca_analysis"):
        with st.spinner("Performing PCA analysis on test images..."):
            pca_analysis()

# ---------------- Feature Maps Page ----------------
def feature_maps_page():
    st.title("🎨 Feature Map Visualization")
    st.markdown("""
    <div style='background-color: #112936; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3>Understanding Feature Maps</h3>
        <p>Feature maps help us visualize what the neural network "sees" in different layers. The figure below is enlarged for clarity.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload MRI scan for feature maps...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("### 🔍 Feature Maps")
            with st.spinner("Generating feature maps..."):
                visualize_feature_maps(image)

# =============================================================================
# Main App Logic
# =============================================================================
def main():
    nav = sidebar_nav()
    if nav == "🏠 Home":
        home_page()
    elif nav == "🔍 Predict":
        predict_page()
    elif nav == "🖼 Advanced Analysis":
        advanced_analysis_page()
    elif nav == "💬 AI Chat":
        ai_chat_page()
    elif nav == "📊 Evaluation":
        evaluation_page()
    elif nav == "🎨 Feature Maps":
        feature_maps_page()

if __name__ == "__main__":
    main()
