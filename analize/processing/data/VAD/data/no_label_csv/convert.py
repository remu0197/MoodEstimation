import os
import shutil

pathes = os.listdir('./')

for path in pathes :
    if not os.path.isdir(path) :
        continue

    files = os.listdir(path)
    for i, file in enumerate(files) :
        new_filename = path + '{:0=3}.csv'.format(i+1)
        shutil.copyfile(path + '/' + file, new_filename)