@echo off
color 0A
echo =======================================================
echo     PROSES UPLOAD KE GITHUB (Decoupled GKHV)
echo =======================================================
echo.

echo 1. Menambahkan perubahan file...
git add .
echo.

echo 2. Membuat rekaman commit otomatis...
git commit -m "Update otomatis: Penyesuaian arsitektur GKHV Frontend & Backend"
echo.

echo 3. Mengunggah ke repository GitHub...
git push origin main

echo.
echo =======================================================
echo     BERHASIL! KODE SUDAH TER-PUSH KE GITHUB
echo =======================================================
echo Silakan cek Vercel dan Render/Railway Anda.
pause
