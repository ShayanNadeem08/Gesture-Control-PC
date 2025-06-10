# -*- mode: python ; coding: utf-8 -*-

import os
import sys
# Search for mediapipe package
for path in sys.path:
    if os.path.isdir(path):
        if "mediapipe" in os.listdir(path):
            mp_path = path+"\\mediapipe\\modules\\"
            break

a = Analysis(
    ['app_integrate.py'],
    pathex=[],
    binaries=[],
    datas=[(mp_path+'\\hand_landmark', 'mediapipe\\modules\\hand_landmark'),
           (mp_path+'\\palm_detection', 'mediapipe\\modules\\palm_detection')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
splash = Splash(
    '../installer/icon.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    [],
    exclude_binaries=True,
    name='app_integrate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app_integrate',
)
