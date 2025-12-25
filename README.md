# 🍽️ Food-11 Image Classification

**Ujian Akhir Praktikum (UAP) – Pembelajaran Mesin**
Universitas Muhammadiyah Malang

---

## 📌 Deskripsi

Aplikasi web berbasis **Streamlit** untuk **klasifikasi citra makanan** ke dalam **11 kategori** menggunakan **Deep Learning** dan **Transfer Learning**.
Dikembangkan sebagai bagian dari **UAP Pembelajaran Mesin** di Laboratorium Informatika UMM.

---

## 👨‍🎓 Identitas Mahasiswa

Nama: **Nur Fitrah Wahyuni**
NIM: **202210370311213**
Kelas: **Pembelajaran Mesin C**
Program Studi: Informatika
Universitas: Universitas Muhammadiyah Malang

---

## 🎯 Tujuan

* Implementasi CNN Non-Pretrained
* Implementasi Transfer Learning (pretrained models)
* Evaluasi dan perbandingan performa model
* Pembuatan web app interaktif dengan Streamlit

---

## 🧠 Model yang Digunakan

* **CNN Base (Non-Pretrained)** – Dibangun dan dilatih dari awal
* **EfficientNetB7 (Pretrained ImageNet)** – Transfer learning dengan fine-tuning
* **MobileNetV2 (Pretrained ImageNet)** – Lightweight, feature extraction

---

## 🏷️ Kelas Dataset (Food-11)

Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit

---

## 📊 Dataset

* **Food-11 Image Dataset (Kaggle)**
* Total: **16.643 gambar**, 11 kelas
* Split: Train / Validation / Test
  [https://www.kaggle.com/datasets/trolukovich/food11-image-dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)

---

## 🖥️ Fitur Aplikasi

* Upload & prediksi gambar makanan
* Multi-model prediction
* Top-5 prediction visualization
* Confusion Matrix & Classification Report
* Model comparison dashboard

---

## ⚙️ Menjalankan Aplikasi

1. `pip install -r requirements.txt`
2. `streamlit run app.py`

---

## 🛠️ Teknologi

TensorFlow, Keras, Streamlit, NumPy, Pandas, Plotly, Python

---

## 📈 Evaluasi

Evaluasi model menggunakan **Accuracy, Precision, Recall, F1-Score**, Confusion Matrix, dan Training History.
Hasil evaluasi tersedia di folder `reports/`.

## 📈 Hasil Evaluasi dan Analisis Perbandingan Model

### Tabel Perbandingan Model

| Nama Model | Test Accuracy | Analisis |
|-----------|:-------------:|----------|
| **CNN Base (Non-Pretrained)** | **74.48%** | Model CNN dasar yang dibangun dari awal, memerlukan training lebih lama |
| **EfficientNetB7 (Pretrained)** | **92.74%** ⭐ | Model pretrained dengan arsitektur kompleks, performa tinggi dengan fine-tuning |
| **MobileNetV2 (Pretrained)** | **87.36%** | Model pretrained yang ringan dan cepat, cocok untuk deployment |

### 📊 Detail Analisis per Model

#### 1. CNN Base (Non-Pretrained) - 74.48%

**Karakteristik:**
- ✅ Dibangun dari awal tanpa pretrained weights
- ✅ Custom architecture untuk Food-11 dataset
- ✅ Total parameters: ~2M (semua trainable)
- ⚠️ Training time lebih lama (~2 jam)
- ⚠️ Membutuhkan data augmentation intensif

**Kelebihan:**
- Full control atas architecture
- Tidak bergantung pada pretrained models
- Cocok untuk pembelajaran fundamental CNN
- Model size relatif kecil (~25 MB)

**Kekurangan:**
- Akurasi paling rendah (74.48%)
- Membutuhkan waktu training yang lama
- Performa kurang optimal dibanding pretrained models

**Use Case:**
- 📚 Learning dan research
- 🎓 Memahami CNN dari basic
- 📊 Baseline comparison

#### 2. EfficientNetB7 (Pretrained) - 92.74% ⭐

**Karakteristik:**
- ✅ Pretrained pada ImageNet (1.4M images)
- ✅ Fine-tuning 30 layer terakhir
- ✅ Total parameters: 66M+ (8M trainable)
- ⚠️ Model size besar (~260 MB)
- ⚠️ Inference time lebih lambat (~2.5s/image)

**Kelebihan:**
- **Akurasi tertinggi: 92.74%** 🏆
- Transfer learning sangat efektif (+18.26% vs CNN Base)
- Robust pada berbagai kondisi gambar
- State-of-the-art architecture

**Kekurangan:**
- Resource intensive (GPU required)
- Model size besar (260 MB)
- Inference time lambat untuk real-time apps

**Use Case:**
- 🏢 Production dengan high accuracy requirement
- 🖥️ Server-side deployment
- 📈 Critical accuracy applications

#### 3. MobileNetV2 (Pretrained) - 87.36%

**Karakteristik:**
- ✅ Pretrained lightweight architecture
- ✅ Frozen base model (feature extraction)
- ✅ Total parameters: 3.5M (400K trainable)
- ✅ Model size kecil (~14 MB)
- ✅ Fast inference (~0.8s/image)

**Kelebihan:**
- **Balance optimal** akurasi vs efisiensi
- 3x lebih cepat dari EfficientNetB7
- Model size 18x lebih kecil (14 MB vs 260 MB)
- Cocok untuk mobile deployment
- Training time cepat (~1 jam)

**Kekurangan:**
- Akurasi 5.38% lebih rendah dari EfficientNetB7

**Use Case:**
- 📱 Mobile applications
- ⚡ Real-time systems
- 🔋 Edge devices
- 💾 Resource-constrained environments

### 🏆 Model Comparison Summary

| Aspek | CNN Base | EfficientNetB7 | MobileNetV2 |
|-------|:--------:|:--------------:|:-----------:|
| **Accuracy** | 74.48% | **92.74%** ⭐ | 87.36% |
| **Model Size** | 25 MB | 260 MB | **14 MB** ⭐ |
| **Inference Time** | 1.2s | 2.5s | **0.8s** ⭐ |
| **Parameters** | 2M | 66M | 3.5M |
| **Training Time** | 2h | 4h | **1h** ⭐ |
| **Pretrained** | ❌ | ✅ | ✅ |
| **Mobile Ready** | ✅ | ❌ | **✅** ⭐ |

### 💡 Kesimpulan

1. **Transfer Learning Impact:**
   - EfficientNetB7: **+18.26%** improvement
   - MobileNetV2: **+12.88%** improvement
   - Transfer learning dari ImageNet sangat efektif

2. **Model Selection:**
   - **High Accuracy Priority** → EfficientNetB7 (92.74%)
   - **Efficiency Priority** → MobileNetV2 (87.36%)
   - **Learning Purpose** → CNN Base (74.48%)

3. **Production Recommendation:**
   - **Server/Cloud**: EfficientNetB7 untuk akurasi maksimal
   - **Mobile/Edge**: MobileNetV2 untuk balance terbaik
   - **Prototype**: CNN Base untuk quick testing

## ✨ Penutup

Project ini diharapkan dapat menjadi media pembelajaran **Image Classification dan Deep Learning**.

📌 **UAP Pembelajaran Mesin – 2025**
👨‍💻 **Nur Fitrah Wahyuni | 202210370311213**
