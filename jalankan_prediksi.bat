@echo off
cd /d C:\Users\lenovo\Documents\project_dss
echo [%date% %time%] MULAI menjalankan uji_prediksi.py >> log_scheduler.txt
py uji_prediksi.py >> log_scheduler.txt 2>&1
echo [%date% %time%] SELESAI >> log_scheduler.txt
echo ---------------------------------------------------------- >> log_scheduler.txt