import requests
import pandas as pd
from datetime import datetime

API_KEY = "569240161c856e676b0a57f38800fff0"
city    = "Tangerang"

url = (
    f"https://api.openweathermap.org/data/2.5/weather"
    f"?q={city}&appid={API_KEY}&units=metric"
)

response = requests.get(url)
data     = response.json()

print(data)

# ============================================================
# VALIDASI API
# ============================================================
if "main" not in data:
    print("API ERROR:", data.get("message", "Unknown error"))
    exit(1)

tanggal        = datetime.now().strftime("%d-%m-%Y")
suhu_rata      = data["main"]["temp"]
kelembapan     = data["main"]["humidity"]
curah_hujan    = data.get("rain", {}).get("1h", 0)
angin          = data["wind"]["speed"]
sunrise        = data["sys"]["sunrise"]
sunset         = data["sys"]["sunset"]
lama_penyinaran = (sunset - sunrise) / 3600

realtime_df = pd.DataFrame([{
    "tanggal"        : tanggal,
    "suhu_rata"      : suhu_rata,
    "kelembapan"     : kelembapan,
    "curah_hujan"    : curah_hujan,
    "lama_penyinaran": lama_penyinaran,
    "angin"          : angin
}])

print(realtime_df)

# ============================================================
# GABUNGKAN DENGAN DATA HISTORIS
# ============================================================
historis = pd.read_csv("Data/jakarta_tangerang.csv")

combined = pd.concat([historis, realtime_df], ignore_index=True)

combined["stasiun"] = combined["stasiun"].fillna(99999)

combined.to_csv("Data/final_dataset.csv", index=False)

print("Dataset berhasil digabung!")
print(combined.tail())