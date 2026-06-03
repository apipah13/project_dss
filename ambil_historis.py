import requests
import pandas as pd

# ============================================================
# KOORDINAT TANGERANG
# ============================================================

latitude = -6.1783
longitude = 106.6319

# ============================================================
# API HISTORICAL OPEN METEO
# ============================================================

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}"
    f"&longitude={longitude}"
    f"&start_date=2005-01-01"
    f"&end_date=2025-12-31"
    f"&daily="
    f"temperature_2m_mean,"
    f"relative_humidity_2m_mean,"
    f"sunshine_duration,"
    f"precipitation_sum,"
    f"wind_speed_10m_max"
    f"&timezone=Asia%2FBangkok"
)

# ============================================================
# REQUEST API
# ============================================================

response = requests.get(url)
data = response.json()

daily = data["daily"]

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame({
    "tanggal": daily["time"],
    "suhu_rata": daily["temperature_2m_mean"],
    "kelembapan": daily["relative_humidity_2m_mean"],

    # detik → jam
    "lama_penyinaran": [
        x / 3600 if x is not None else None
        for x in daily["sunshine_duration"]
    ],

    "curah_hujan": daily["precipitation_sum"],
    "angin": daily["wind_speed_10m_max"]
})

# ============================================================
# SIMPAN CSV
# ============================================================

df.to_csv("tangerang_2005_2025.csv", index=False)

print("✅ Dataset berhasil disimpan!")
print(df.head())

df = df.dropna()
df = df.drop_duplicates()

df["tanggal"] = pd.to_datetime(df["tanggal"])

df["bulan"] = df["tanggal"].dt.month
df["hari"] = df["tanggal"].dt.day

df["target_hujan"] = (df["curah_hujan"] > 0).astype(int)
df["suhu_kemarin"] = df["suhu_rata"].shift(1)
df["kelembapan_kemarin"] = df["kelembapan"].shift(1)
df = df.dropna()