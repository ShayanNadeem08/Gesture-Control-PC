# Remove unrequired libraries from bundle prepared by pyinstaller

import os

f = open("installer/lib_list.txt")
allowed_lib_list = f.read().split("\n")
f.close()

print("Removing unrequired libraries:")
present_lib_list = os.listdir("app_integrate/_internal/")
for file in present_lib_list:
    if file not in allowed_lib_list:
        print(file)
        if os.path.isdir("app_integrate/_internal/"+file):
            os.rmdir("app_integrate/_internal/"+file)
        else:
            os.remove("app_integrate/_internal/"+file)