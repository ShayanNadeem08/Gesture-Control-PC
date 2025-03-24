# Gesture-Control-PC
A deep learning-based real-time hand gesture recognition system optimized for gaming applications. This project enhances user experience by replacing traditional input devices with natural hand movements.

### Usage
Train: Train_model.ipynb\
Capture data: Capture_dataset.ipynb\
Use to control applications: run app_integrate.py

### Files description
#### run:
Contains scripts to run the product.\
**clean.bat**: Clears the build files generated when running.

#### codes:
Contains files used to train model.

**Capture_dataset.ipynb**: Code to capture dataset as video frames.\
**Train_model.ipynb**: Code to train and test model.\
**app_integrate.py**: App main file: Run it to using model for controlling applications.\
**gesture_recog.py**: App file containing gesture recogination code.\
**model.py**: App file defining DENSENET model.\
**key_map.py**: App file for taking actions based on gestures.

#### external_modules:
Contains components that are taken from sources from internet and used in the project.\
**video_dataset.py**: Used to import dataset.

#### model:
Contains saved trained model. Each time a model is trained, it is saved with its own timestamp in its name.

#### results:
Contains logs and plots generated during training.

