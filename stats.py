""" 
get the numerical stats of your game using cv2 and mss
"""
import time

import cv2
import mss
import numpy as np
from PIL import Image

from winlaunch import current_windows, win_desktop, win_name, win_pos, win_size

import os
from pathlib import Path

DEBUG = True
HERE = Path(__file__).parent.absolute()

os.environ['TESSDATA_PREFIX'] = str(HERE.joinpath('tessdata/').absolute())

class legWindow:
    def __init__(self, wid):
        self.wid = wid
        self.height, self.height = tuple(win_size(wid))
        self.monitor = win_desktop(wid)+1


# https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# main
for window in current_windows():
    name = win_name(window)
    if name == 'League of Legends (TM) Client' or 'League of Legends' in name:
        winX, winY = tuple(win_size(window))
        print(tuple(win_pos(window)))
        winMonitor = win_desktop(window)+1
        print(f'{name}: {winX}, {winY}, monitor {winMonitor}')


        with mss.mss() as sct:
            while True:
                start = time.perf_counter()
                mon = sct.monitors[winMonitor]

                # The screen part to capture
                monitor = {
                    "top": mon["top"],
                    "left": mon["left"],
                    "width": winX,
                    "height": winY,
                    "mon": winMonitor,
                }

                # Grab the data
                img = np.array(sct.grab(monitor))
                imgResize = resizeWithAspectRatio(img, width=900)
                imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
                # this is good for object detection
                imgCanny = cv2.Canny(imgResize, 100, 150)

                # this is good for text
                # thresh = cv2.adaptiveThreshold(imgGray, 255, 1, 1, 11, 2)
                roi = img[mon["top"]:50, mon["left"]+550:mon["left"]+620]
                # clock = pytesseract.image_to_string(Image.fromarray(roi)).replace('\n', '').strip()
                # print(clock)

                cv2.imshow('the name? rango 2', imgCanny)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                end = time.perf_counter()
                print(f'frame took {round(end-start, 3)} seconds')

    else:
        print('league client not found, exiting')