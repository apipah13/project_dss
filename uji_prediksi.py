import joblib
import pandas as pd
import requests
from datetime import datetime

# ============================================================
# LOAD MODEL
# ============================================================
model = joblib.load("model_cuaca_rf.pkl")

print("✅ Model berhasil diload!")
print("=" * 60)

# ============================================================
# KOORDINAT LOKASI
# Contoh: Tangerang
# ============================================================
latitude = -6.1783
longitude = 106.6319

# ============================================================
# AMBIL DATA DARI OPEN-METEO API
# ============================================================
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={latitude}&longitude={longitude}"
    f"&daily=temperature_2m_mean,"
    f"relative_humidity_2m_mean,"
    f"sunshine_duration,"
    f"precipitation_sum,"
    f"wind_speed_10m_max"
    f"&timezone=Asia%2FBangkok"
    f"&past_days=2"
)

response = requests.get(url)
data = response.json()

# ============================================================
# AMBIL DATA DAILY
# ============================================================
daily = data["daily"]

suhu = daily["temperature_2m_mean"]
kelembapan = daily["relative_humidity_2m_mean"]
penyinaran = daily["sunshine_duration"]
hujan = daily["precipitation_sum"]
angin = daily["wind_speed_10m_max"]

# ============================================================
# INDEX:
# 0 = 2 hari lalu
# 1 = kemarin
# 2 = hari ini
# ============================================================

today = datetime.now()

# ============================================================
# DATA HARI INI
# ============================================================
suhu_rata = suhu[2]
kelembapan_hari_ini = kelembapan[2]

# konversi detik → jam
lama_penyinaran = penyinaran[2] / 3600

kecepatan_angin = angin[2]

bulan = today.month
hari = today.day

# ============================================================
# DATA KEMARIN
# ============================================================
suhu_kemarin = suhu[1]
kelembapan_kemarin = kelembapan[1]

# jika hujan > 0 mm maka dianggap hujan
hujan_kemarin = 1 if hujan[1] > 0 else 0

# ============================================================
# DATA 2 HARI LALU
# ============================================================
suhu_2halu = suhu[0]
kelembapan_2halu = kelembapan[0]

# ============================================================
# BUAT DATAFRAME
# ============================================================
data_input = pd.DataFrame([{
    "suhu_rata": suhu_rata,
    "kelembapan": kelembapan_hari_ini,
    "lama_penyinaran": lama_penyinaran,
    "angin": kecepatan_angin,
    "bulan": bulan,
    "hari": hari,
    "suhu_kemarin": suhu_kemarin,
    "kelembapan_kemarin": kelembapan_kemarin,
    "hujan_kemarin": hujan_kemarin,
    "suhu_2halu": suhu_2halu,
    "kelembapan_2halu": kelembapan_2halu,
}])

# ============================================================
# PREDIKSI
# ============================================================
prediksi = model.predict(data_input)[0]
probabilitas = model.predict_proba(data_input)[0]

prob_tidak_hujan = probabilitas[0] * 100
prob_hujan = probabilitas[1] * 100

# ============================================================
# TAMPILKAN INPUT DATA
# ============================================================
print("📋 DATA CUACA REALTIME")
print("=" * 60)

print(f"📅 Tanggal              : {today.strftime('%d-%m-%Y')}")
print(f"🌡️ Suhu rata-rata       : {suhu_rata:.2f} °C")
print(f"💧 Kelembapan           : {kelembapan_hari_ini:.2f} %")
print(f"☀️ Lama penyinaran      : {lama_penyinaran:.2f} jam")
print(f"💨 Kecepatan angin      : {kecepatan_angin:.2f} m/s")
print(f"🌧️ Hujan kemarin        : {'Ya' if hujan_kemarin == 1 else 'Tidak'}")

print(f"🌡️ Suhu kemarin         : {suhu_kemarin:.2f} °C")
print(f"💧 Kelembapan kemarin   : {kelembapan_kemarin:.2f} %")

print(f"🌡️ Suhu 2 hari lalu     : {suhu_2halu:.2f} °C")
print(f"💧 Kelembapan 2 hari lalu : {kelembapan_2halu:.2f} %")

# ============================================================
# HASIL PREDIKSI
# ============================================================
print("\n🎯 HASIL PREDIKSI")
print("=" * 60)

print(
    f"Prediksi Cuaca : "
    f"{'🌧️ HUJAN' if prediksi == 1 else '☀️ TIDAK HUJAN'}"
)

print(f"Probabilitas Hujan       : {prob_hujan:.2f}%")
print(f"Probabilitas Tidak Hujan : {prob_tidak_hujan:.2f}%")

# ============================================================
# TINGKAT KEYAKINAN MODEL
# ============================================================
print("\n📊 TINGKAT KEYAKINAN MODEL")
print("=" * 60)

if prob_hujan >= 75:
    print("⚠️ Sangat yakin HUJAN")

elif prob_hujan >= 55:
    print("🌦️ Kemungkinan besar HUJAN")

elif prob_tidak_hujan >= 75:
    print("✅ Sangat yakin TIDAK HUJAN")

elif prob_tidak_hujan >= 55:
    print("🌤️ Kemungkinan besar TIDAK HUJAN")

else:
    print("🤔 Model kurang yakin")

# ============================================================
# REKOMENDASI WAKTU TANAM
# ============================================================
print("\n🌱 REKOMENDASI TANAM")
print("=" * 60)

if prediksi == 1:
    print("⚠️ Disarankan MENUNDA penanaman.")
    print("Karena potensi hujan cukup tinggi.")

else:
    print("✅ Waktu tanam cukup baik.")
    print("Cuaca diprediksi tidak hujan.")

print("=" * 60)

hasil = pd.DataFrame([{
    "tanggal": today.strftime("%d-%m-%Y"),
    "prediksi": "HUJAN" if prediksi == 1 else "TIDAK HUJAN",
    "prob_hujan": round(prob_hujan, 2),
    "prob_tidak_hujan": round(prob_tidak_hujan, 2),
    "rekomendasi": (
        "Tunda Tanam"
        if prediksi == 1
        else "Waktu Tanam Baik"
    )
}])

import os

# ============================================================
# SIMPAN HISTORI PREDIKSI
# ============================================================

import os

file_csv = "laporan_prediksi.csv"

# tambah timestamp lengkap
hasil["tanggal"] = today.strftime("%d-%m-%Y %H:%M:%S")

# kalau file sudah ada
if os.path.exists(file_csv):

    # baca data lama
    df_lama = pd.read_csv(file_csv)

    # gabungkan data lama + baru
    df_gabung = pd.concat([df_lama, hasil], ignore_index=True)

    # hapus duplicate
    df_gabung = df_gabung.drop_duplicates()

    # simpan lagi
    df_gabung.to_csv(file_csv, index=False)

else:

    # buat file baru
    hasil.to_csv(file_csv, index=False)

print("\n✅ Histori prediksi berhasil disimpan!")


# ============================================================
# GRAFIK DSS CUACA
# ============================================================

import matplotlib.pyplot as plt

# membuat 1 figure dengan 2 grafik
fig, ax = plt.subplots(1, 2, figsize=(12,5))

# ============================================================
# PIE CHART
# ============================================================

labels = ['Hujan', 'Tidak Hujan']
sizes = [prob_hujan, prob_tidak_hujan]

ax[0].pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

ax[0].set_title("Probabilitas Prediksi Cuaca")

# ============================================================
# BAR CHART
# ============================================================

fitur = [
    'Suhu',
    'Kelembapan',
    'Penyinaran',
    'Angin'
]

nilai = [
    suhu_rata,
    kelembapan_hari_ini,
    lama_penyinaran,
    kecepatan_angin
]

ax[1].bar(fitur, nilai)

ax[1].set_title("Kondisi Cuaca Hari Ini")

ax[1].set_ylabel("Nilai")

# ============================================================
# TAMPILKAN SEMUA GRAFIK
# ============================================================

plt.tight_layout()

plt.show()