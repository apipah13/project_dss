import joblib
import pandas as pd
import requests
from datetime import datetime
import os
import matplotlib.pyplot as plt

# ============================================================
# TEST LOG
# ============================================================
with open("test_log.txt", "a") as f:
    f.write(
        f"Program berjalan - "
        f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
    )

# ============================================================
# LOAD MODEL
# ============================================================
model = joblib.load("model_cuaca_rf.pkl")

print("Model berhasil diload!")
print("=" * 60)

try:

    # ============================================================
    # KOORDINAT LOKASI - Tangerang
    # ============================================================
    latitude  = -6.1783
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
        f"&timezone=Asia%2FJakarta"
        f"&past_days=2"
    )

    response = requests.get(url)
    data = response.json()

    # ============================================================
    # VALIDASI DATA
    # ============================================================
    daily = data["daily"]

    suhu       = daily["temperature_2m_mean"]
    kelembapan = daily["relative_humidity_2m_mean"]
    penyinaran = daily["sunshine_duration"]
    hujan      = daily["precipitation_sum"]
    angin      = daily["wind_speed_10m_max"]

    if len(suhu) < 3:
        raise ValueError(
            f"Data API tidak cukup. Hanya ada {len(suhu)} hari"
        )

    # ============================================================
    # FITUR WAKTU
    # ============================================================
    tanggal_hari_ini = daily["time"][2]
    today = datetime.strptime(tanggal_hari_ini, "%Y-%m-%d")

    bulan = today.month
    hari  = today.day

    # ============================================================
    # DATA CUACA
    # ============================================================
    suhu_rata             = suhu[2]
    kelembapan_hari_ini   = kelembapan[2]
    lama_penyinaran       = penyinaran[2] / 3600
    kecepatan_angin       = angin[2]

    suhu_kemarin          = suhu[1]
    kelembapan_kemarin    = kelembapan[1]
    hujan_kemarin         = 1 if hujan[1] > 0 else 0

    suhu_2hari_lalu       = suhu[0]
    kelembapan_2hari_lalu = kelembapan[0]

    # ============================================================
    # DATAFRAME INPUT
    # ============================================================
    data_input = pd.DataFrame([{
        "suhu_rata"          : suhu_rata,
        "kelembapan"         : kelembapan_hari_ini,
        "lama_penyinaran"    : lama_penyinaran,
        "angin"              : kecepatan_angin,
        "bulan"              : bulan,
        "hari"               : hari,
        "suhu_kemarin"       : suhu_kemarin,
        "kelembapan_kemarin" : kelembapan_kemarin,
        "hujan_kemarin"      : hujan_kemarin,
        "suhu_2halu"         : suhu_2hari_lalu,
        "kelembapan_2halu"   : kelembapan_2hari_lalu,
    }])

    # ============================================================
    # PREDIKSI
    # ============================================================
    prediksi     = model.predict(data_input)[0]
    probabilitas = model.predict_proba(data_input)[0]

    prob_tidak_hujan = probabilitas[0] * 100
    prob_hujan       = probabilitas[1] * 100

    # ============================================================
    # PENENTUAN STATUS CUACA
    # ============================================================
    if prob_hujan >= 75:

        status_cuaca = "HUJAN"
        rekomendasi = "Tunda Tanam"

    elif prob_hujan >= 60:

        status_cuaca = "BERPOTENSI HUJAN"
        rekomendasi = "Pertimbangkan Menunda"

    else:

        status_cuaca = "TIDAK HUJAN"
        rekomendasi = "Waktu Tanam Baik"

    # ============================================================
    # TAMPILKAN HASIL
    # ============================================================
    print("\n")
    print("=" * 60)
    print("DATA CUACA REALTIME")
    print("=" * 60)

    print(f"Waktu                    : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    print(f"Suhu rata-rata           : {suhu_rata:.2f} C")
    print(f"Kelembapan               : {kelembapan_hari_ini:.2f} %")
    print(f"Lama penyinaran          : {lama_penyinaran:.2f} jam")
    print(f"Kecepatan angin          : {kecepatan_angin:.2f} m/s")

    print("\nHASIL PREDIKSI")
    print("=" * 60)

    print(f"Status Cuaca             : {status_cuaca}")
    print(f"Probabilitas Hujan       : {prob_hujan:.2f}%")
    print(f"Probabilitas Tidak Hujan : {prob_tidak_hujan:.2f}%")

    # ============================================================
    # TINGKAT KEYAKINAN MODEL
    # ============================================================
    print("\nTINGKAT KEYAKINAN MODEL")
    print("=" * 60)

    if prob_hujan >= 85:

        print("Model sangat yakin akan terjadi hujan.")

    elif prob_hujan >= 75:

        print("Kemungkinan hujan tinggi.")

    elif prob_hujan >= 60:

        print("Ada potensi hujan.")

    else:

        print("Cuaca cenderung aman.")

    # ============================================================
    # REKOMENDASI TANAM
    # ============================================================
    print("\nREKOMENDASI TANAM")
    print("=" * 60)

    if status_cuaca == "HUJAN":

        print("Disarankan MENUNDA penanaman.")
        print("Karena potensi hujan sangat tinggi.")

    elif status_cuaca == "BERPOTENSI HUJAN":

        print("Cuaca berpotensi hujan.")
        print("Sebaiknya mempertimbangkan kondisi lapangan.")

    else:

        print("Waktu tanam cukup baik.")
        print("Cuaca diprediksi tidak hujan.")

    # ============================================================
    # SIMPAN HISTORI KE CSV
    # ============================================================
    timestamp_sekarang = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    hasil = pd.DataFrame([{
        "tanggal"          : timestamp_sekarang,
        "prediksi"         : status_cuaca,
        "prob_hujan"       : round(prob_hujan, 2),
        "prob_tidak_hujan" : round(prob_tidak_hujan, 2),
        "rekomendasi"      : rekomendasi
    }])

    file_csv = "laporan_prediksi.csv"

    if os.path.exists(file_csv):

        hasil.to_csv(
            file_csv,
            mode='a',
            header=False,
            index=False
        )

    else:

        hasil.to_csv(
            file_csv,
            index=False
        )

    print("\nHistori prediksi berhasil disimpan!")

    # ============================================================
    # GRAFIK DSS CUACA
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # PIE CHART
    labels = ["Hujan", "Tidak Hujan"]
    sizes  = [prob_hujan, prob_tidak_hujan]

    ax[0].pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%"
    )

    ax[0].set_title("Probabilitas Prediksi Cuaca")

    # BAR CHART
    fitur = [
        "Suhu",
        "Kelembapan",
        "Penyinaran",
        "Angin"
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

    plt.tight_layout()

    # ============================================================
    # SIMPAN GRAFIK
    # ============================================================
    nama_grafik = (
        f"grafik_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

    plt.savefig(nama_grafik)

    plt.close()

    print(f"Grafik berhasil disimpan: {nama_grafik}")

    print("=" * 60)
    print("PROGRAM SELESAI")
    print("=" * 60)

except Exception as e:

    print(f"Terjadi error: {e}")

    with open("error_log.txt", "a") as f:

        f.write(
            f"{datetime.now()} - {str(e)}\n"
        )

    print("Error berhasil disimpan ke error_log.txt")