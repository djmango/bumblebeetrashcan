""" 
get the numerical stats of your game using cv2 and mss
"""
import time

import cv2

import mss
import numpy as np
from PIL import Image

from winlaunch import current_windows, win_desktop, win_name, win_pos, win_size, press_key

import os
from pathlib import Path

DEBUG = True
HERE = Path(__file__).parent.absolute()

os.environ['TESSDATA_PREFIX'] = str(HERE.joinpath('tessdata/').absolute())

class legWindow:
    """ a class specific to league game window, contains window info and utils """
    def __init__(self, wid):
        # simple geometry
        self.wid = wid
        self.name = win_name(wid)
        self.width, self.height = tuple(win_size(wid))
        self.x, self.y = tuple(win_pos(wid))

        # determine if on left or right monitor. this is a bad method, only works iwth my two monitor setup
        if self.x >= 1920: # right monitor
            monitor = 1
        else: # left monitor
            monitor = 2

        # window dimensions into mss format
        self.geometry = {
            "top": self.y,
            "left": self.x,
            "width": self.width,
            "height": self.height,
            "mon": monitor
        }
        # to get window position relative to a monitor we have to account for monitor position
        self.x += -1920

    def wper(self, width):
        """ takes in width in percent and returns calculated pixel value based on screen dimensions """
        return int((self.width/100*width)+self.x) # essentially this calc is getting relative window x in the first set of parentheses and then making them absolute with the addition of the abs x

    def hper(self, height):
        """ takes in height in percent and returns calculated pixel value based on screen dimensions """
        return int((self.height/100*height)+self.y)

class legVideo:
    """ a class with cv2 video info and utils """
    def __init__(self, videoPath):
        # simple geometry
        self.capture = cv2.VideoCapture(videoPath)
        self.width  = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

    def wper(self, width):
        """ takes in width in percent and returns calculated pixel value based on video dimensions """
        return int(self.width/100*width)

    def hper(self, height):
        """ takes in height in percent and returns calculated pixel value based on video dimensions """
        return int(self.height/100*height)

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
def main():
    for i, window in enumerate(current_windows()):
        name = win_name(window)
        if name == 'League of Legends (TM) Client' or 'League of Legends' in name:
            window = legWindow(window)

            with mss.mss() as sct:
                while True:
                    start = time.perf_counter()

                    # Grab the screen
                    img = np.array(sct.grab(window.geometry))
                    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # this is good for object detection
                    imgCanny = cv2.Canny(img, 100, 150)

                    # this is good for text
                    thresh = cv2.adaptiveThreshold(imgGray, 255, 1, 1, 11, 2)
                    roi = thresh[window.hper(.5):window.hper(2.5), window.wper(96.7):window.wper(99)]
                    # roi = img[mon["top"]:50, mon["left"]+550:mon["left"]+620]
                    # clock = pytesseract.image_to_string(Image.fromarray(roi)).replace('\n', '').strip()
                    # print(clock)
                    
                    cv2.imshow('the name? rango 2', resizeWithAspectRatio(roi, width=900))
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

                    end = time.perf_counter()
                    print(f'frame took {round(end-start, 3)} seconds')

        elif i == len(current_windows()):
            print('league client not found, exiting')

def captureVid(videoName):
    videoPath = str(HERE.joinpath('traindata', 'basev', videoName))
    video = legVideo(videoPath)
    frame = 0
    frameLast = 0
    secondsBuffer = 69 # ex 60 if we start from 1 min

    ptime = time.perf_counter()
    while True:
        success, img = video.capture.read()
        if frame == frameLast + 60:
            frameLast = frame
            roi = img[video.hper(.5):video.hper(2.5), video.wper(96.7):video.wper(99.1)]
            cv2.imshow('vid', roi)
            cv2.imwrite(str(HERE.joinpath('traindata', 'clock', f'{round(frameLast/60)+secondsBuffer}.png')), roi)

            print(frame)
            print(f'second took {round(time.perf_counter()-ptime, 3)} seconds')
            ptime = time.perf_counter()

        frame=frame+1
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

# capture traindata
def capture():
    for i, window in enumerate(current_windows()):
        name = win_name(window)
        if name == 'League of Legends (TM) Client' or 'League of Legends' in name:
            window = legWindow(window)

            with mss.mss() as sct:
                k = 69

                ptime = time.perf_counter()
                while True:
                    img = np.array(sct.grab(window.geometry))
                    roi = img[window.hper(.5):window.hper(2.5), window.wper(96.7):window.wper(99)]
                    cv2.imwrite(str(HERE.joinpath('traindata', 'clock', f'{k}.png')), roi)
                    
                    # mss.tools.to_png(img.rgb, img.size, output=HERE.joinpath('traindata', 'clock', f'{k}.png'))
                    while ptime+1 >= time.perf_counter():
                        time.sleep(.001)
                    ptime = time.perf_counter()
                    k=k+1
                    print(f'{k}: {time.perf_counter()}')

        elif i == len(current_windows()):
            print('league client not found, exiting')

if __name__ == '__main__':
    captureVid('foxdropgraves-bXdBOIqujbs.mkv')