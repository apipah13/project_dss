import joblib
import pandas as pd
import requests
from datetime import datetime
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI (mudah diubah untuk eksperimen)
# ============================================================
# Nilai default (dipakai kalau threshold_config.json belum ada,
# misal sebelum train_model.py versi baru dijalankan).
THRESHOLD_HUJAN            = 75   # >= ini -> status HUJAN
THRESHOLD_BERPOTENSI_HUJAN = 60   # >= ini -> status BERPOTENSI HUJAN

# ------------------------------------------------------------
# PERUBAHAN: baca threshold hasil kalibrasi otomatis dari
# train_model.py (threshold_config.json), kalau tersedia.
# Ini memastikan uji_prediksi.py selalu pakai threshold yang
# sudah dioptimalkan berdasarkan data terbaru, bukan angka
# hardcoded yang bisa jadi sudah tidak sesuai.
# ------------------------------------------------------------
if os.path.exists("threshold_config.json"):
    with open("threshold_config.json", "r") as f:
        cfg = json.load(f)
    THRESHOLD_HUJAN = cfg["threshold_hujan_persen"]
    THRESHOLD_BERPOTENSI_HUJAN = max(THRESHOLD_HUJAN - 15, 40)
    print(f"Threshold dimuat dari threshold_config.json: {THRESHOLD_HUJAN}%")
    print(f"  (Macro F1 kalibrasi: {cfg.get('macro_f1_pada_threshold')}, "
          f"ambang curah hujan: {cfg.get('ambang_curah_hujan_mm')} mm)")
else:
    print("threshold_config.json tidak ditemukan, pakai threshold default "
          f"({THRESHOLD_HUJAN}%). Jalankan train_model.py versi terbaru "
          "untuk kalibrasi otomatis.")
print("=" * 60)

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

# ------------------------------------------------------------
# PERUBAHAN: cek fitur apa saja yang dipakai saat training,
# supaya kita tahu urutan/nama kolom yang benar.
# ------------------------------------------------------------
if hasattr(model, "feature_names_in_"):
    print("Fitur yang diharapkan model (urutan saat training):")
    print(list(model.feature_names_in_))
    print("=" * 60)
else:
    print("PERINGATAN: model tidak menyimpan feature_names_in_.")
    print("Model mungkin dilatih dari array numpy, bukan DataFrame.")
    print("Urutan kolom input HARUS sama persis dengan urutan saat training.")
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
    data     = response.json()

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
    # PERINGATAN: index [2] adalah HARI INI, dan karena hari belum
    # selesai, data suhu/kelembapan/penyinaran untuk hari ini dari
    # Open-Meteo bisa jadi masih PARSIAL (rata-rata dari jam yang
    # sudah lewat saja, bukan rata-rata 24 jam penuh). Ini bisa
    # membuat kelembapan tampak lebih tinggi dari kondisi
    # sebenarnya, terutama kalau script dijalankan malam hari
    # (kelembapan malam umumnya lebih tinggi dari siang).
    # Pertimbangkan pakai index [1] (kemarin, data sudah lengkap)
    # untuk prediksi yang lebih stabil, atau jalankan script di
    # jam yang konsisten tiap hari.
    # ============================================================

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

    # ------------------------------------------------------------
    # PERUBAHAN PENTING: paksa urutan kolom input sama persis
    # dengan urutan kolom saat model dilatih. Ini mencegah bug
    # "silent" di mana RandomForest salah membaca fitur karena
    # urutan kolom berbeda dari training.
    # ------------------------------------------------------------
    if hasattr(model, "feature_names_in_"):
        kolom_model = list(model.feature_names_in_)
        kolom_input = list(data_input.columns)

        if set(kolom_model) != set(kolom_input):
            hilang  = set(kolom_model) - set(kolom_input)
            ekstra  = set(kolom_input) - set(kolom_model)
            raise ValueError(
                f"Kolom input tidak cocok dengan model!\n"
                f"Kolom hilang di input : {hilang}\n"
                f"Kolom ekstra di input : {ekstra}"
            )

        if kolom_model != kolom_input:
            print("PERINGATAN: urutan kolom input berbeda dari training.")
            print(f"  Urutan training : {kolom_model}")
            print(f"  Urutan input    : {kolom_input}")
            print("  -> Kolom input akan diurutkan ulang otomatis.\n")

        data_input = data_input[kolom_model]

    # ============================================================
    # DEBUG: tampilkan nilai fitur yang dipakai untuk prediksi
    # ============================================================
    print("Fitur yang digunakan untuk prediksi kali ini:")
    print(data_input.to_string(index=False))
    print()

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
    if prob_hujan >= THRESHOLD_HUJAN:
        status_cuaca = "HUJAN"
        rekomendasi  = "Tunda Tanam"

    elif prob_hujan >= THRESHOLD_BERPOTENSI_HUJAN:
        status_cuaca = "BERPOTENSI HUJAN"
        rekomendasi  = "Pertimbangkan Menunda"

    else:
        status_cuaca = "TIDAK HUJAN"
        rekomendasi  = "Waktu Tanam Baik"

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
    elif prob_hujan >= THRESHOLD_HUJAN:
        print("Kemungkinan hujan tinggi.")
    elif prob_hujan >= THRESHOLD_BERPOTENSI_HUJAN:
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
        hasil.to_csv(file_csv, mode="a", header=False, index=False)
    else:
        hasil.to_csv(file_csv, index=False)

    print("\nHistori prediksi berhasil disimpan!")

    # ============================================================
    # GRAFIK DSS CUACA
    # ============================================================
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DSS Prediksi Cuaca Tangerang - "
        f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        fontsize=14,
        fontweight="bold"
    )

    # ============================================================
    # 1. PIE CHART - Probabilitas
    # ============================================================
    labels = ["Hujan", "Tidak Hujan"]
    sizes  = [prob_hujan, prob_tidak_hujan]
    colors = ["#2196F3", "#FFC107"]

    ax[0, 0].pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )
    ax[0, 0].set_title("Probabilitas Prediksi Cuaca")

    # ============================================================
    # 2. GAUGE CHART - Speedometer Probabilitas Hujan
    # ============================================================
    gauge_val  = prob_hujan / 100
    theta      = np.linspace(0, np.pi, 300)

    ax[0, 1].plot(
        np.cos(theta), np.sin(theta),
        color="lightgrey", linewidth=15
    )

    theta_fill = np.linspace(0, np.pi * gauge_val, 300)

    if prob_hujan >= THRESHOLD_HUJAN:
        gauge_color = "#F44336"
    elif prob_hujan >= THRESHOLD_BERPOTENSI_HUJAN:
        gauge_color = "#FF9800"
    else:
        gauge_color = "#4CAF50"

    ax[0, 1].plot(
        np.cos(theta_fill), np.sin(theta_fill),
        color=gauge_color, linewidth=15
    )

    angle = np.pi * gauge_val
    ax[0, 1].annotate(
        "",
        xy=(np.cos(angle) * 0.7, np.sin(angle) * 0.7),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2)
    )

    ax[0, 1].text(
        0, -0.3,
        f"{prob_hujan:.1f}%",
        ha="center", fontsize=16, fontweight="bold",
        color=gauge_color
    )
    ax[0, 1].text(
        0, -0.55,
        status_cuaca,
        ha="center", fontsize=11, fontweight="bold",
        color=gauge_color
    )

    ax[0, 1].set_xlim(-1.3, 1.3)
    ax[0, 1].set_ylim(-0.7, 1.2)
    ax[0, 1].axis("off")
    ax[0, 1].set_title("Gauge Probabilitas Hujan")
    ax[0, 1].text(-1.1, -0.1, "0%",   ha="center", fontsize=9, color="grey")
    ax[0, 1].text( 1.1, -0.1, "100%", ha="center", fontsize=9, color="grey")
    ax[0, 1].text( 0,    1.1, "50%",  ha="center", fontsize=9, color="grey")

    # ============================================================
    # 3. BAR CHART - Kelembapan 3 Hari
    # ============================================================
    hari_label   = ["2 Hari Lalu", "Kemarin", "Hari Ini"]
    nilai_lembap = [
        kelembapan_2hari_lalu,
        kelembapan_kemarin,
        kelembapan_hari_ini
    ]
    bar_colors = ["#90CAF9", "#42A5F5", "#1565C0"]

    bars = ax[1, 0].bar(
        hari_label, nilai_lembap,
        color=bar_colors, edgecolor="white"
    )
    ax[1, 0].set_title("Perbandingan Kelembapan 3 Hari")
    ax[1, 0].set_ylabel("Kelembapan (%)")
    ax[1, 0].set_ylim(0, 110)

    for bar, val in zip(bars, nilai_lembap):
        ax[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", fontsize=10, fontweight="bold"
        )

    # ============================================================
    # 4. TABEL RINGKASAN HASIL PREDIKSI
    # ============================================================
    ax[1, 1].axis("off")

    tabel_data = [
        ["Tanggal",            datetime.now().strftime("%d-%m-%Y")],
        ["Waktu",              datetime.now().strftime("%H:%M:%S")],
        ["Suhu Rata-rata",     f"{suhu_rata:.2f} C"],
        ["Kelembapan",         f"{kelembapan_hari_ini:.2f} %"],
        ["Lama Penyinaran",    f"{lama_penyinaran:.2f} jam"],
        ["Kecepatan Angin",    f"{kecepatan_angin:.2f} m/s"],
        ["Status Cuaca",       status_cuaca],
        ["Prob. Hujan",        f"{prob_hujan:.2f}%"],
        ["Prob. Tidak Hujan",  f"{prob_tidak_hujan:.2f}%"],
        ["Rekomendasi",        rekomendasi],
    ]

    tabel = ax[1, 1].table(
        cellText=tabel_data,
        colLabels=["Parameter", "Nilai"],
        cellLoc="center",
        loc="center"
    )
    tabel.auto_set_font_size(False)
    tabel.set_fontsize(10)
    tabel.scale(1.2, 1.6)

    for j in range(2):
        tabel[0, j].set_facecolor("#1565C0")
        tabel[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(tabel_data):
        if row[0] == "Status Cuaca":
            if status_cuaca == "HUJAN":
                tabel[i + 1, 1].set_facecolor("#FFCDD2")
            elif status_cuaca == "BERPOTENSI HUJAN":
                tabel[i + 1, 1].set_facecolor("#FFE0B2")
            else:
                tabel[i + 1, 1].set_facecolor("#C8E6C9")

    ax[1, 1].set_title("Ringkasan Hasil Prediksi")

    plt.tight_layout()

    # ============================================================
    # SIMPAN GRAFIK
    # ============================================================
    nama_grafik = (
        f"grafik_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

    plt.savefig(nama_grafik, dpi=150, bbox_inches="tight")
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