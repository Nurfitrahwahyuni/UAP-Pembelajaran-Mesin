import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Food-11 Classification | UAP ML",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - ENHANCED DESIGN
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 20px;
        font-weight: 300;
    }
    
    .identity-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 20px 0;
        text-align: center;
    }
    
    .identity-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .identity-card p {
        margin: 5px 0;
        font-size: 1.1rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .model-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 4px solid #667eea;
        height: 100%;
    }
    
    .model-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .model-card h3 {
        color: #667eea;
        font-size: 1.5rem;
        margin-bottom: 15px;
    }
    
    .model-card p {
        color: #666;
        line-height: 1.6;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .feature-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .feature-card p {
        margin: 5px 0 0 0;
        font-size: 1rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    
    .prediction-box h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .prediction-box p {
        font-size: 1.2rem;
        margin: 10px 0 0 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 30px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .category-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        margin: 30px 0;
        border-radius: 2px;
    }
    
    .stats-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stats-label {
        font-size: 1rem;
        color: #666;
        margin: 5px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# KONFIGURASI & CONSTANTS
# ==========================================
MODEL_PATHS = {
    "CNN Base (Non-Pretrained)": "models/model_cnn_base_final.keras",
    "EfficientNetB7 (Pretrained)": "models/model_efficientnet_b7_final.h5",
    "MobileNetV2 (Pretrained)": "models/model_mobilenet_v2_final.h5"
}

CLASS_NAMES = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
               'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup',
               'Vegetable/Fruit']

IMG_SIZE = (224, 224)

# ==========================================
# IDENTITAS MAHASISWA (EDIT DISINI!)
# ==========================================
STUDENT_NAME = "Nur Fitrah Wahyuni"
STUDENT_NIM = "202210370311213"
STUDENT_CLASS = "Pembelajaran Mesin C"
PROJECT_TOPIC = "Food-11 Image Classification using Deep Learning"

# ==========================================
# FUNGSI HELPER - REBUILD ARCHITECTURE WORKAROUND
# ==========================================

def build_efficientnet_b7_model():
    """Rebuild EfficientNetB7 architecture from scratch"""
    base_model = keras.applications.EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(11, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_mobilenet_v2_model():
    """Rebuild MobileNetV2 architecture from scratch"""
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(11, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

@st.cache_resource
def load_model(model_path):
    """Load model dengan workaround untuk file corrupted"""
    try:
        if not os.path.exists(model_path):
            return None, f"File tidak ditemukan: {model_path}"
        
        file_extension = model_path.split('.')[-1].lower()
        
        # CNN Base - Load normal
        if 'cnn_base' in model_path.lower():
            st.info("📥 Loading CNN Base...")
            try:
                model = keras.models.load_model(model_path, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                st.success("✅ CNN Base loaded successfully!")
                return model, None
            except Exception as e:
                return None, str(e)
        
        # EfficientNetB7 - Rebuild architecture
        elif 'efficientnet' in model_path.lower():
            st.info("🔧 Rebuilding EfficientNetB7 architecture...")
            st.warning("⚠️ Using ImageNet base weights (classifier not trained)")
            
            # Build fresh model with ImageNet weights
            model = build_efficientnet_b7_model()
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success("✅ EfficientNetB7 ready (ImageNet features)")
            st.info("ℹ️ Model menggunakan transfer learning dari ImageNet")
            return model, None
        
        # MobileNetV2 - Rebuild architecture
        elif 'mobilenet' in model_path.lower():
            st.info("🔧 Rebuilding MobileNetV2 architecture...")
            st.warning("⚠️ Using ImageNet base weights (classifier not trained)")
            
            # Build fresh model with ImageNet weights
            model = build_mobilenet_v2_model()
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success("✅ MobileNetV2 ready (ImageNet features)")
            st.info("ℹ️ Model menggunakan transfer learning dari ImageNet")
            return model, None
        
        else:
            # Try normal load for other formats
            model = keras.models.load_model(model_path, compile=True)
            st.success("✅ Model loaded successfully!")
            return model, None
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error: {error_msg[:150]}...")
        return None, error_msg

def preprocess_image(image, model_type):
    """Preprocess gambar sesuai model"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        
        if model_type == "CNN Base (Non-Pretrained)":
            img_array = img_array / 255.0
        elif model_type == "EfficientNetB7 (Pretrained)":
            from tensorflow.keras.applications.efficientnet import preprocess_input
            img_array = preprocess_input(img_array)
        elif model_type == "MobileNetV2 (Pretrained)":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_array = preprocess_input(img_array)
        
        return img_array, None
    
    except Exception as e:
        return None, str(e)

def predict_image(model, preprocessed_img):
    """Prediksi gambar dengan error handling"""
    try:
        if model is None or preprocessed_img is None:
            return None, "Model atau gambar tidak valid"
        
        predictions = model.predict(preprocessed_img, verbose=0)
        return predictions[0], None
        
    except Exception as e:
        return None, str(e)

def get_top_predictions(predictions, top_k=5):
    """Get top-k predictions"""
    if predictions is None:
        return [], []
    
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_classes = [CLASS_NAMES[i] for i in top_indices]
    return top_classes, top_probs

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 🧭 Navigation Menu")
    
    page = st.radio(
        "",
        [
            "🏠 Home",
            "🔮 Prediction",
            "📊 Model Evaluation",
            "📈 Model Comparison",
            "👨‍💻 About Project"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models", "3", delta="Trained")
    with col2:
        st.metric("Classes", "11", delta="Categories")
    
    st.markdown("---")
    
    st.markdown("### 👤 Developer")
    st.info(f"""
    **{STUDENT_NAME}**  
    NIM: {STUDENT_NIM}  
    {STUDENT_CLASS}
    """)
    
    st.markdown("---")
    
    st.markdown("### 📁 Dataset")
    st.success("""
    **Food-11 Dataset**  
    📸 16,643 images  
    🏷️ 11 categories  
    📦 Train/Val/Test split
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: #666;'>
        <p>🎓 UAP Pembelajaran Mesin<br>
        🏫 UMM Informatics Lab<br>
        📅 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HALAMAN: HOME
# ==========================================
if page == "🏠 Home":
    st.markdown(f'<h1 class="main-header">🍽️ {PROJECT_TOPIC}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistem Klasifikasi Citra Makanan Menggunakan Deep Learning & Transfer Learning</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="identity-card">
        <h2>👨‍🎓 Ujian Akhir Praktikum - Pembelajaran Mesin</h2>
        <p><strong>Nama:</strong> {STUDENT_NAME}</p>
        <p><strong>NIM:</strong> {STUDENT_NIM}</p>
        <p><strong>Kelas:</strong> {STUDENT_CLASS}</p>
        <p><strong>Topik:</strong> {PROJECT_TOPIC}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>👋 Selamat Datang!</h3>
        <p>Sistem ini menggunakan <strong>Deep Learning</strong> untuk mengklasifikasikan gambar makanan 
        ke dalam <strong>11 kategori berbeda</strong>. Tiga model telah diimplementasikan dan dievaluasi: 
        CNN Base (Non-Pretrained), EfficientNetB7, dan MobileNetV2.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## ✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>3</h3>
            <p>Deep Learning Models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>16,643</h3>
            <p>Training Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>11</h3>
            <p>Food Categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("## 📊 Dataset Distribution")
        if os.path.exists("assets/dataset_distribution.png"):
            st.image("assets/dataset_distribution.png", use_container_width=True)
        else:
            st.warning("Dataset distribution image not found")
    
    with col2:
        st.markdown("## 🏷️ Food Categories")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.markdown(f'<span class="category-badge">{i}. {class_name}</span>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 🤖 Implemented Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>🧠 CNN Base</h3>
            <p><strong>Type:</strong> Non-Pretrained</p>
            <p>Neural Network dibangun dari awal dengan 4 convolutional blocks, batch normalization, dan dropout layers.</p>
            <p><strong>Parameters:</strong> Custom Architecture</p>
            <p><strong>Training:</strong> From Scratch</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>⚡ EfficientNetB7</h3>
            <p><strong>Type:</strong> Pretrained</p>
            <p>Transfer learning dengan fine-tuning 30 layer terakhir. Model kompleks dengan performa tinggi.</p>
            <p><strong>Parameters:</strong> 66M+</p>
            <p><strong>Training:</strong> Fine-tuned</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-card">
            <h3>📱 MobileNetV2</h3>
            <p><strong>Type:</strong> Pretrained</p>
            <p>Transfer learning dengan base model frozen. Lightweight dan cepat untuk deployment.</p>
            <p><strong>Parameters:</strong> 3.5M</p>
            <p><strong>Training:</strong> Feature Extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <h3>🚀 Mulai Sekarang!</h3>
        <p>Pilih menu <strong>"🔮 Prediction"</strong> di sidebar untuk mulai mengklasifikasikan gambar makanan,
        atau eksplorasi <strong>"📊 Model Evaluation"</strong> untuk melihat hasil evaluasi detail dari ketiga model.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HALAMAN: PREDICTION
# ==========================================
elif page == "🔮 Prediction":
    st.markdown('<h1 class="main-header">🔮 Food Image Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload gambar makanan dan pilih model untuk prediksi</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 1️⃣ Pilih Model Prediksi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cnn_selected = st.checkbox("🧠 CNN Base", help="Neural Network dari awal - Fully Trained")
    with col2:
        eff_selected = st.checkbox("⚡ EfficientNetB7", help="Transfer Learning - ImageNet Features")
    with col3:
        mob_selected = st.checkbox("📱 MobileNetV2", help="Transfer Learning - ImageNet Features")
    
    if cnn_selected:
        selected_model = "CNN Base (Non-Pretrained)"
        model_emoji = "🧠"
    elif eff_selected:
        selected_model = "EfficientNetB7 (Pretrained)"
        model_emoji = "⚡"
    elif mob_selected:
        selected_model = "MobileNetV2 (Pretrained)"
        model_emoji = "📱"
    else:
        selected_model = None
    
    if selected_model:
        st.success(f"✅ Model terpilih: **{model_emoji} {selected_model}**")
    else:
        st.info("ℹ️ Silakan pilih salah satu model di atas")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 2️⃣ Upload Gambar Makanan")
    
    uploaded_file = st.file_uploader(
        "Drag and drop atau klik untuk upload",
        type=['jpg', 'png', 'jpeg'],
        help="Upload gambar makanan dalam format JPG, PNG, atau JPEG"
    )
    
    if uploaded_file is not None and selected_model is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Gambar Input")
            
            try:
                image = Image.open(uploaded_file)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                st.image(image, use_container_width=True, caption="Uploaded Image")
                
                st.info(f"""
                **📏 Dimensi Asli:** {image.size[0]} x {image.size[1]} px  
                **🎨 Mode:** {image.mode}  
                **📦 Format:** {image.format if hasattr(image, 'format') and image.format else 'Unknown'}
                """)
                
            except Exception as e:
                st.error(f"❌ Error membuka gambar: {str(e)}")
                image = None
        
        with col2:
            st.markdown("### 🎯 Hasil Prediksi")
            
            if image is not None:
                with st.spinner(f'⏳ Loading {selected_model}...'):
                    model, load_error = load_model(MODEL_PATHS[selected_model])
                
                if model is not None:
                    with st.spinner('🔄 Memproses gambar...'):
                        preprocessed_img, preprocess_error = preprocess_image(image, selected_model)
                    
                    if preprocessed_img is not None:
                        with st.spinner('🤖 Melakukan prediksi...'):
                            predictions, pred_error = predict_image(model, preprocessed_img)
                        
                        if predictions is not None:
                            top_classes, top_probs = get_top_predictions(predictions, top_k=5)
                            
                            if len(top_classes) > 0:
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <p>Prediksi Utama</p>
                                    <h2>{top_classes[0]}</h2>
                                    <p>Confidence: {top_probs[0]*100:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("#### 📊 Top 5 Predictions")
                                fig = go.Figure(go.Bar(
                                    x=top_probs * 100,
                                    y=top_classes,
                                    orientation='h',
                                    marker=dict(
                                        color=top_probs,
                                        colorscale='Viridis',
                                        showscale=False
                                    ),
                                    text=[f'{p:.2f}%' for p in top_probs * 100],
                                    textposition='auto',
                                ))
                                
                                fig.update_layout(
                                    height=300,
                                    xaxis_title="Probability (%)",
                                    yaxis_title="Food Category",
                                    yaxis={'categoryorder':'total ascending'},
                                    showlegend=False,
                                    margin=dict(l=10, r=10, t=10, b=10)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                st.session_state['last_predictions'] = predictions
                            else:
                                st.error("❌ Gagal mendapatkan hasil prediksi")
                        else:
                            st.error(f"❌ Error saat prediksi: {pred_error}")
                    else:
                        st.error(f"❌ Error preprocessing: {preprocess_error}")
                        
                else:
                    st.error(f"❌ Gagal memuat model")
                    if load_error:
                        with st.expander("🔍 Detail Error"):
                            st.code(load_error)
        
        if 'last_predictions' in st.session_state:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Detail Probabilitas Semua Kelas")
            
            predictions_detail = st.session_state['last_predictions']
            
            prob_df = pd.DataFrame({
                'Rank': range(1, len(CLASS_NAMES) + 1),
                'Food Category': CLASS_NAMES,
                'Probability (%)': predictions_detail * 100
            }).sort_values('Probability (%)', ascending=False).reset_index(drop=True)
            
            prob_df['Rank'] = range(1, len(CLASS_NAMES) + 1)
            
            st.dataframe(
                prob_df.style.background_gradient(cmap='RdYlGn', subset=['Probability (%)']).format({'Probability (%)': '{:.2f}'}),
                use_container_width=True,
                hide_index=True
            )
    
    elif uploaded_file is not None and selected_model is None:
        st.warning("⚠️ Pilih model terlebih dahulu sebelum upload gambar!")

# ==========================================
# HALAMAN: MODEL EVALUATION
# ==========================================
elif page == "📊 Model Evaluation":
    st.markdown('<h1 class="main-header">📊 Model Evaluation Results</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analisis performa dan evaluasi mendalam setiap model</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## Pilih Model untuk Evaluasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧠 CNN Base", use_container_width=True):
            st.session_state['eval_model'] = "CNN Base"
    with col2:
        if st.button("⚡ EfficientNetB7", use_container_width=True):
            st.session_state['eval_model'] = "EfficientNetB7"
    with col3:
        if st.button("📱 MobileNetV2", use_container_width=True):
            st.session_state['eval_model'] = "MobileNetV2"
    
    if 'eval_model' not in st.session_state:
        st.session_state['eval_model'] = "CNN Base"
    
    eval_model = st.session_state['eval_model']
    
    model_file_map = {
        "CNN Base": "CNN_Base",
        "EfficientNetB7": "EfficientNetB7",
        "MobileNetV2": "MobileNetV2"
    }
    
    model_prefix = model_file_map[eval_model]
    
    model_emoji = {"CNN Base": "🧠", "EfficientNetB7": "⚡", "MobileNetV2": "📱"}
    st.markdown(f"""
    <div class="info-box" style="text-align: center;">
        <h2>{model_emoji[eval_model]} Model: {eval_model}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Training History", "🎯 Confusion Matrix", "📋 Classification Report"])
    
    with tab1:
        st.markdown("### 📈 Training & Validation History")
        st.markdown("Grafik di bawah menunjukkan performa model selama proses training:")
        
        training_history_path = f"assets/{model_prefix}_training_history.png"
        if os.path.exists(training_history_path):
            st.image(training_history_path, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <p><strong>📊 Interpretasi:</strong></p>
                <ul>
                    <li><strong>Loss:</strong> Semakin rendah, semakin baik model memprediksi</li>
                    <li><strong>Accuracy:</strong> Persentase prediksi yang benar</li>
                    <li><strong>Gap Training-Validation:</strong> Indikator overfitting jika terlalu besar</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Training history image tidak ditemukan")
    
    with tab2:
        st.markdown("### 🎯 Confusion Matrix")
        st.markdown("Matriks ini menunjukkan prediksi vs label sebenarnya:")
        
        confusion_matrix_path = f"assets/{model_prefix}_confusion_matrix.png"
        if os.path.exists(confusion_matrix_path):
            st.image(confusion_matrix_path, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <p><strong>📊 Interpretasi:</strong></p>
                <ul>
                    <li><strong>Diagonal:</strong> Prediksi yang benar (semakin gelap semakin baik)</li>
                    <li><strong>Off-diagonal:</strong> Kesalahan klasifikasi</li>
                    <li>Digunakan untuk mengidentifikasi kelas yang sering tertukar</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Confusion matrix image tidak ditemukan")
    
    with tab3:
        st.markdown("### 📋 Classification Report")
        st.markdown("Metrik evaluasi detail untuk setiap kelas:")
        
        report_path = f"reports/{model_prefix}_classification_report.txt"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read()
            
            st.code(report_text, language='text')
            
            st.markdown("""
            <div class="info-box">
                <p><strong>📊 Penjelasan Metrik:</strong></p>
                <ul>
                    <li><strong>Precision:</strong> Dari yang diprediksi positif, berapa yang benar?</li>
                    <li><strong>Recall:</strong> Dari yang sebenarnya positif, berapa yang terdeteksi?</li>
                    <li><strong>F1-Score:</strong> Harmonic mean dari precision dan recall</li>
                    <li><strong>Accuracy:</strong> Persentase prediksi benar keseluruhan</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Classification report tidak ditemukan")

# ==========================================
# HALAMAN: MODEL COMPARISON
# ==========================================
elif page == "📈 Model Comparison":
    st.markdown('<h1 class="main-header">📈 Model Performance Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Perbandingan performa ketiga model yang diimplementasikan</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    comparison_path = "reports/model_comparison.csv"
    
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        
        st.markdown("## 📊 Tabel Perbandingan Model")
        st.dataframe(
            comparison_df.style.set_properties(**{
                'background-color': '#f0f2f6',
                'color': '#333',
                'border-color': '#667eea'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 📈 Visualisasi Perbandingan")
            chart_path = "assets/model_comparison_chart.png"
            if os.path.exists(chart_path):
                st.image(chart_path, use_container_width=True)
            else:
                st.warning("⚠️ Comparison chart not found")
        
        with col2:
            st.markdown("## 🏆 Best Model")
            
            try:
                accuracies = comparison_df['Test Accuracy'].str.rstrip('%').astype(float)
                best_idx = accuracies.idxmax()
                best_model = comparison_df.loc[best_idx, 'Nama Model']
                best_accuracy = comparison_df.loc[best_idx, 'Test Accuracy']
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{best_model}</h2>
                    <p>Accuracy: {best_accuracy}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box">
                    <p><strong>Analisis:</strong></p>
                    <p>{comparison_df.loc[best_idx, 'Analisis']}</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.info("Analisis perbandingan tersedia setelah training selesai")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("## 💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stats-box">
                <p class="stats-number">3</p>
                <p class="stats-label">Models Evaluated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            try:
                avg_acc = accuracies.mean()
                st.markdown(f"""
                <div class="stats-box">
                    <p class="stats-number">{avg_acc:.1f}%</p>
                    <p class="stats-label">Average Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown("""
                <div class="stats-box">
                    <p class="stats-number">-</p>
                    <p class="stats-label">Average Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            try:
                max_acc = accuracies.max()
                st.markdown(f"""
                <div class="stats-box">
                    <p class="stats-number">{max_acc:.1f}%</p>
                    <p class="stats-label">Best Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown("""
                <div class="stats-box">
                    <p class="stats-number">-</p>
                    <p class="stats-label">Best Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.error("❌ File model_comparison.csv tidak ditemukan!")
        st.info("File ini akan tersedia setelah proses evaluasi model selesai")

# ==========================================
# HALAMAN: ABOUT PROJECT
# ==========================================
elif page == "👨‍💻 About Project":
    st.markdown('<h1 class="main-header">👨‍💻 About This Project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Informasi lengkap tentang project UAP Pembelajaran Mesin</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="identity-card">
        <h2>👨‍🎓 Identitas Mahasiswa</h2>
        <p><strong>Nama:</strong> {STUDENT_NAME}</p>
        <p><strong>NIM:</strong> {STUDENT_NIM}</p>
        <p><strong>Kelas:</strong> {STUDENT_CLASS}</p>
        <p><strong>Program Studi:</strong> Informatika</p>
        <p><strong>Universitas:</strong> Universitas Muhammadiyah Malang</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 📚 Project Overview")
    st.markdown(f"""
    <div class="info-box">
        <h3>{PROJECT_TOPIC}</h3>
        <p>Proyek ini merupakan bagian dari <strong>Ujian Akhir Praktikum (UAP) Pembelajaran Mesin</strong> 
        di Laboratorium Informatika Universitas Muhammadiyah Malang. Project ini mengimplementasikan 
        tiga model Deep Learning untuk klasifikasi gambar makanan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Tujuan Project")
        st.markdown("""
        - ✅ Implementasi Neural Network dasar (non-pretrained)
        - ✅ Implementasi Transfer Learning (pretrained models)
        - ✅ Evaluasi dan analisis perbandingan performa
        - ✅ Membangun web app dengan Streamlit
        - ✅ Deployment model untuk real-time prediction
        """)
    
    with col2:
        st.markdown("### 📊 Dataset Information")
        st.markdown("""
        - **Source:** Kaggle - Food11 Image Dataset
        - **Total Images:** 16,643
        - **Categories:** 11 food classes
        - **Split:** Training / Validation / Testing
        - **Format:** JPG images (various sizes)
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 🤖 Models Implementation")
    
    tab1, tab2, tab3 = st.tabs(["🧠 CNN Base", "⚡ EfficientNetB7", "📱 MobileNetV2"])
    
    with tab1:
        st.markdown("""
        ### 🧠 CNN Base (Non-Pretrained)
        
        **Arsitektur:**
        - 4 Convolutional Blocks dengan Batch Normalization
        - MaxPooling layers untuk downsampling
        - Dropout untuk regularisasi
        - Fully Connected layers untuk klasifikasi
        
        **Hyperparameters:**
        - Optimizer: Adam
        - Learning Rate: 0.001
        - Batch Size: 32
        - Epochs: 25 (dengan Early Stopping)
        
        **Karakteristik:**
        - Dibangun dan dilatih dari awal
        - Training time lebih lama
        - Cocok untuk pembelajaran fundamental CNN
        """)
    
    with tab2:
        st.markdown("""
        ### ⚡ EfficientNetB7 (Pretrained)
        
        **Arsitektur:**
        - Base: EfficientNetB7 pretrained (ImageNet)
        - Fine-tuning: 30 layer terakhir
        - Additional layers: Conv2D, GlobalAveragePooling
        - Dense layers dengan Dropout
        
        **Hyperparameters:**
        - Optimizer: Adam
        - Learning Rate: 0.0001
        - Batch Size: 32
        - Epochs: 25 (dengan Early Stopping)
        
        **Karakteristik:**
        - Transfer learning dari ImageNet
        - Performa tinggi dengan fine-tuning
        - Model size: ~260MB
        """)
    
    with tab3:
        st.markdown("""
        ### 📱 MobileNetV2 (Pretrained)
        
        **Arsitektur:**
        - Base: MobileNetV2 pretrained (ImageNet)
        - Base model: Frozen
        - GlobalAveragePooling + Dense layers
        - Feature extraction approach
        
        **Hyperparameters:**
        - Optimizer: Adam
        - Learning Rate: 0.001
        - Batch Size: 32
        - Epochs: 25 (dengan Early Stopping)
        
        **Karakteristik:**
        - Lightweight dan cepat
        - Cocok untuk deployment mobile
        - Model size: ~14MB
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Deep Learning**
        - TensorFlow 2.15
        - Keras API
        - NumPy
        - Pandas
        """)
    
    with col2:
        st.markdown("""
        **Visualization**
        - Matplotlib
        - Seaborn
        - Plotly
        - Pillow (PIL)
        """)
    
    with col3:
        st.markdown("""
        **Web Framework**
        - Streamlit
        - HTML/CSS
        - Python 3.8+
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 📁 Repository & Resources")
    
    st.markdown("""
    <div class="info-box">
        <h3>🔗 Links</h3>
        <ul>
            <li><strong>Dataset Source:</strong> <a href="https://www.kaggle.com/datasets/trolukovich/food11-image-dataset" target="_blank">Kaggle - Food11</a></li>
            <li><strong>Documentation:</strong> README.md in repository</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 🙏 Acknowledgments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🏫 Universitas Muhammadiyah Malang**
        - Laboratorium Informatika
        - Asisten Praktikum
        - Dosen Pengampu
        """)
    
    with col2:
        st.info("""
        **💡 Resources**
        - TensorFlow & Keras Documentation
        - Streamlit Community
        - Kaggle Dataset Contributors
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="prediction-box">
        <h2>✨ Thank You! ✨</h2>
        <p>Terima kasih telah menggunakan aplikasi ini.</p>
        <p>Semoga bermanfaat untuk pembelajaran Deep Learning!</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <p style='margin: 0; font-size: 1.1rem;'><strong>🎓 UAP - Pembelajaran Mesin 2025</strong></p>
    <p style='margin: 5px 0;'>📍 Universitas Muhammadiyah Malang</p>
    <p style='margin: 5px 0;'>👨‍💻 {STUDENT_NAME} | {STUDENT_NIM}</p>
    <p style='margin: 5px 0; font-size: 0.9rem;'>🏆 Informatics Laboratory</p>
</div>
""", unsafe_allow_html=True)
