# UAP-Pembelajaran-Mesin
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

---

## ✨ Penutup

Project ini diharapkan dapat menjadi media pembelajaran **Image Classification dan Deep Learning**.

📌 **UAP Pembelajaran Mesin – 2025**
👨‍💻 **Nur Fitrah Wahyuni | 202210370311213**
