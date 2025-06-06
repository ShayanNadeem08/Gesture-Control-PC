# Gesture-Control-PC
A deep learning-based real-time hand gesture recognition system optimized for gaming applications. This project enhances user experience by replacing traditional input devices with natural hand movements.

### Run (code)
app/app.exe is the executable. Run it by double clicking or run one of the following commands in this directory:

Run: 
```
./run/gui
```
Run without gui: 
```
./run/app
```
Train: Train_model.ipynb\
Capture data: Capture_dataset.ipynb\

### Install
Run installer/Setup.exe.

### Build installer
To create installer to allow distribution of system, first use pyinstaller to build an executable from python code:
```
./run/build
```

Then use InstallForge to generate Setup.exe. (Settings are already done in installer/installer.ifp. Just open it and click build.)

### Files description
#### run:
Contains scripts to run the product.\
**run.bat**: Runs system.\
**clean.bat**: Clears the build files generated when running.

#### app:
Contains source and execuateble files for gui apps.\
Source code: app.sb\
Execuatable code: app.exe\

#### codes:
Contains files used to train model.

#### code_dataset:
Contains files used to generate dataset.

#### code_old:
Contains old codes tried previously during development.

#### installer:
Contains InstallForge setup and files to build installer.

#### model:
Contains trained model(s).

#### output:
Contains logs and plots generated during training.
