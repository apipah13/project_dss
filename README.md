# DSS Prediksi Cuaca Tangerang

Sistem prediksi cuaca otomatis menggunakan Machine Learning (Random Forest)
untuk rekomendasi waktu tanam.

## Cara Install

1. Install Python 3.9 → https://python.org
2. Buka CMD di folder project, ketik:
   pip install -r requirements.txt

## Cara Jalankan

### Manual
python uji_prediksi.py

### Otomatis (tiap 1 jam)
1. Buka Task Scheduler Windows
2. Buat task baru
3. Arahkan ke file jalankan_prediksi.bat
4. Set repeat every 1 hour

## Struktur File
- uji_prediksi.py → script utama prediksi
- model_cuaca_rf.pkl → model machine learning
- laporan_prediksi.csv → histori hasil prediksi
- jalankan_prediksi.bat → script otomasi Windows