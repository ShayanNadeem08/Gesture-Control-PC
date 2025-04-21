:: Remove pycache and build files

@rmdir /S /Q "code/__pycache__"
@rmdir /S /Q "code/.ipynb_checkpoints"
@rmdir /S /Q "code/debug_frames"
@rmdir /S /Q "code/processed_frames"