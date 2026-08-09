"""
Dashboard DSS Prediksi Cuaca & Waktu Tanam Hidroponik — versi Streamlit
Fitur: prediksi real-time, riwayat prediksi otomatis (auto-log berkala) tersimpan ke CSV, styling colorful.
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ============================================================
# KONFIGURASI HALAMAN & STYLING
# ============================================================
st.set_page_config(
    page_title="DSS Cuaca & Waktu Tanam",
    page_icon="🌦️",
    layout="wide",
)

AMBANG_HUJAN_MM = 1.0  # samakan dengan train_model.py
RIWAYAT_CSV = "riwayat_prediksi_streamlit.csv"

# Interval auto-log (dalam detik). Ubah sesuai kebutuhan, misal 60 = tiap 1 menit.
INTERVAL_AUTO_LOG_DETIK = 60

# --- CSS custom biar nggak polos putih ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 6%, #f1f5f9 6%);
    }
    .header-banner {
        background: linear-gradient(90deg, #1B2452 0%, #2C4BC7 100%);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 18px rgba(27,36,82,0.35);
    }
    .header-banner h1 {
        color: white;
        margin: 0;
        font-size: 28px;
    }
    .header-banner p {
        color: #C7D2FE;
        margin: 4px 0 0 0;
        font-size: 14px;
    }
    .metric-card {
        border-radius: 14px;
        padding: 18px 20px;
        color: white;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }
    .metric-card .label {
        font-size: 13px;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 26px;
        font-weight: 700;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & THRESHOLD
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("model_cuaca_rf.pkl")
    if os.path.exists("threshold_config.json"):
        with open("threshold_config.json", "r") as f:
            cfg = json.load(f)
        threshold_hujan = cfg["threshold_hujan_persen"]
    else:
        threshold_hujan = 40
    threshold_berpotensi = max(threshold_hujan - 15, 40)
    return model, threshold_hujan, threshold_berpotensi


model, THRESHOLD_HUJAN, THRESHOLD_BERPOTENSI_HUJAN = load_model()


# ============================================================
# AMBIL DATA CUACA REAL-TIME
# ============================================================
@st.cache_data(ttl=3600)
def ambil_data_cuaca():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=-6.1783&longitude=106.6319"
        "&daily=temperature_2m_mean,relative_humidity_2m_mean,"
        "sunshine_duration,precipitation_sum,wind_speed_10m_max"
        "&past_days=2&timezone=Asia%2FJakarta"
    )
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def buat_fitur(data_api):
    daily = data_api["daily"]
    suhu = daily["temperature_2m_mean"]
    lembap = daily["relative_humidity_2m_mean"]
    sinar = daily["sunshine_duration"]
    hujan = daily["precipitation_sum"]
    angin = daily["wind_speed_10m_max"]
    now = datetime.now()

    data_input = pd.DataFrame([{
        "suhu_rata": suhu[-1],
        "kelembapan": lembap[-1],
        "lama_penyinaran": sinar[-1],
        "angin": angin[-1],
        "bulan": now.month,
        "hari": now.day,
        "suhu_kemarin": suhu[-2],
        "kelembapan_kemarin": lembap[-2],
        "hujan_kemarin": 1 if hujan[-2] >= AMBANG_HUJAN_MM else 0,
        "suhu_2halu": suhu[-3],
        "kelembapan_2halu": lembap[-3],
    }])
    return data_input, lembap


# ============================================================
# SIMPAN & BACA RIWAYAT PREDIKSI
# ============================================================
def simpan_riwayat(status_cuaca, prob_hujan, prob_tidak_hujan, rekomendasi):
    baris_baru = pd.DataFrame([{
        "waktu": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "status_cuaca": status_cuaca,
        "prob_hujan": round(prob_hujan, 2),
        "prob_tidak_hujan": round(prob_tidak_hujan, 2),
        "rekomendasi": rekomendasi,
    }])
    if os.path.exists(RIWAYAT_CSV):
        baris_baru.to_csv(RIWAYAT_CSV, mode="a", header=False, index=False)
    else:
        baris_baru.to_csv(RIWAYAT_CSV, index=False)


def baca_riwayat():
    if os.path.exists(RIWAYAT_CSV):
        return pd.read_csv(RIWAYAT_CSV)
    return pd.DataFrame(columns=["waktu", "status_cuaca", "prob_hujan", "prob_tidak_hujan", "rekomendasi"])


# ============================================================
# PROSES PREDIKSI
# ============================================================
data_api = ambil_data_cuaca()
data_input, riwayat_kelembapan = buat_fitur(data_api)

probabilitas = model.predict_proba(data_input)[0]
prob_tidak_hujan = probabilitas[0] * 100
prob_hujan = probabilitas[1] * 100

if prob_hujan >= THRESHOLD_HUJAN:
    status_cuaca, rekomendasi, warna, emoji = "HUJAN", "Tunda Tanam", "#E63946", "🔴"
elif prob_hujan >= THRESHOLD_BERPOTENSI_HUJAN:
    status_cuaca, rekomendasi, warna, emoji = "BERPOTENSI HUJAN", "Pertimbangkan Menunda", "#F4A261", "🟠"
else:
    status_cuaca, rekomendasi, warna, emoji = "TIDAK HUJAN", "Waktu Tanam Baik", "#2A9D8F", "🟢"

# --------------------------------------------------------------
# AUTO-LOG BERKALA (bukan cuma pas status berubah)
# --------------------------------------------------------------
# Karena st_autorefresh bikin app rerun tiap INTERVAL_AUTO_LOG_DETIK detik,
# setiap rerun yang terjadi otomatis (bukan karena klik tombol manual) akan
# dicatat sebagai baris riwayat baru — persis kayak kebiasaan log_scheduler.txt lama.
if "log_counter" not in st.session_state:
    st.session_state.log_counter = 0

# st_autorefresh mengembalikan counter yang bertambah tiap kali dia trigger rerun
refresh_count = st_autorefresh(interval=INTERVAL_AUTO_LOG_DETIK * 1000, key="auto_refresh_riwayat")

if refresh_count != st.session_state.log_counter:
    simpan_riwayat(status_cuaca, prob_hujan, prob_tidak_hujan, rekomendasi)
    st.session_state.log_counter = refresh_count


# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="header-banner">
    <h1>🌦️ DSS Prediksi Cuaca &amp; Waktu Tanam Hidroponik</h1>
    <p>Wilayah: Tangerang, Banten &nbsp;|&nbsp; Model: Random Forest &nbsp;|&nbsp; Update terakhir: {datetime.now().strftime('%d %B %Y, %H:%M')} WIB &nbsp;|&nbsp; Auto-log tiap {INTERVAL_AUTO_LOG_DETIK} detik</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# KARTU STATUS (colorful)
# ============================================================
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="metric-card" style="background:{warna};">
        <div class="label">STATUS CUACA</div>
        <div class="value">{emoji} {status_cuaca}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card" style="background:#264653;">
        <div class="label">PROBABILITAS HUJAN</div>
        <div class="value">{prob_hujan:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card" style="background:#1B2452;">
        <div class="label">REKOMENDASI</div>
        <div class="value">{rekomendasi}</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ============================================================
# CHART
# ============================================================
colA, colB = st.columns(2)

with colA:
    st.subheader("📊 Distribusi Probabilitas")
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    fig1.patch.set_alpha(0)
    ax1.pie(
        [prob_hujan, prob_tidak_hujan],
        labels=["Hujan", "Tidak Hujan"],
        autopct="%1.1f%%",
        colors=["#E63946", "#2A9D8F"],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    st.pyplot(fig1)

with colB:
    st.subheader("💧 Tren Kelembapan 3 Hari Terakhir")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_alpha(0)
    bars = ax2.bar(["H-2", "H-1", "Hari ini"], riwayat_kelembapan[-3:], color=["#264653", "#2A9D8F", "#2C4BC7"])
    ax2.set_ylabel("Kelembapan (%)")
    ax2.bar_label(bars, fmt="%.0f%%")
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)

st.write("")
st.subheader("📋 Ringkasan Parameter Cuaca Saat Ini")
st.dataframe(data_input, use_container_width=True, hide_index=True)

st.write("")

# ============================================================
# RIWAYAT PREDIKSI + TOMBOL SAVE
# ============================================================
st.subheader("🕒 Riwayat Prediksi")

col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    if st.button("💾 Simpan Prediksi Sekarang"):
        simpan_riwayat(status_cuaca, prob_hujan, prob_tidak_hujan, rekomendasi)
        st.success("Tersimpan ke riwayat!")

df_riwayat = baca_riwayat()

with col_btn2:
    if not df_riwayat.empty:
        st.download_button(
            "⬇️ Download Riwayat (CSV)",
            data=df_riwayat.to_csv(index=False).encode("utf-8"),
            file_name="riwayat_prediksi.csv",
            mime="text/csv",
        )

if df_riwayat.empty:
    st.info("Belum ada riwayat prediksi tersimpan.")
else:
    st.dataframe(
        df_riwayat.sort_index(ascending=False),
        use_container_width=True,
        hide_index=True,
        height=280,  # tinggi tetap (px) -> kalau isi lebih panjang, otomatis muncul scrollbar
    )

st.write("")
if st.button("🔄 Refresh Data Cuaca dari API"):
    st.cache_data.clear()
    st.rerun()