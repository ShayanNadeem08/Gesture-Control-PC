:: Remove pycache and build files
@rmdir /S /Q __pycache__
@rmdir /S /Q .ipynb_checkpoints
@rmdir /S /Q "code/__pycache__"
@rmdir /S /Q "code/.ipynb_checkpoints"
@rmdir /S /Q "code/debug_frames"
@rmdir /S /Q "code/processed_frames"
@rmdir /S /Q "output"
@rmdir /S /Q "app_integrate"