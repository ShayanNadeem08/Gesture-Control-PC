cd code
::pyinstaller app_integrate.py --clean --noconfirm --splash ../installer/icon.png --distpath ../ --add-data=C:\Users\PMLS\AppData\Roaming\Python\Python312\site-packages\mediapipe\modules\hand_landmark:mediapipe\modules\hand_landmark
pyinstaller app_integrate.spec --clean --noconfirm --distpath ../
cd ..
python installer/clean.py
@echo --
@echo Build complete
@echo Build dir: ../app_integrate