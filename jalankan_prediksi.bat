@echo off
cd /d "C:\Users\lenovo\Documents\project_dss"
set PYTHONIOENCODING=utf-8
"C:\Users\lenovo\AppData\Local\Programs\Python\Python39\python.exe" uji_prediksi.py >> log_scheduler.txt 2>&1
echo [%date% %time%] Script selesai >> log_scheduler.txt