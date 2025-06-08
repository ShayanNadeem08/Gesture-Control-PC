cd code
pyinstaller app_integrate.py --clean --noconfirm --noconsole --splash ../installer/icon.png --distpath ../ --add-data=C:\Users\PMLS\AppData\Roaming\Python\Python312\site-packages\mediapipe\modules:mediapipe\modules\
cd ..
python installer/clean.py