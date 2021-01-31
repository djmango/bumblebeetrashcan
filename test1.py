import cv2
import mss
import time
import numpy as np
from winlaunch import current_windows, win_name, win_size

# mss.init()
# img show
# img = cv2.imread('fl.png')
# cv2.imshow('the image', img)
# cv2.waitKey(0)

# vid
# cap = cv2.VideoCapture('b.mp4')
# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     cv2.imshow('vid', cv2.Canny(img, 150, 200))
#     if cv2.waitKey(10) & 0xFF==ord('q'):
#         break

# img = cv2.imread('fl.png')

# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
# imgCanny = cv2.Canny(img, 150, 200)

# cv2.imshow('greay', imgGray)
# cv2.imshow('gblur', imgBlur)
# cv2.imshow('edge', imgCanny)
# cv2.waitKey(0)

# with mss() as sct:
#     monitor = {"top": 160, "left": 160, "width": 160, "height": 135}
#     while True:
#         last_time = time.time()
#         img = sct.grab(monitor)
#         # print('fps: {0}'.format(1 / (time.time()-last_time)))
#         cv2.imshow('test', np.array(img))
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

with mss.mss() as sct:
    # Get information of monitor 2
    monitor_number = 1
    mon = sct.monitors[monitor_number]

    # The screen part to capture
    monitor = {
        "top": mon["top"] + 100,  # 100px from the top
        "left": mon["left"] + 100,  # 100px from the left
        "width": 160,
        "height": 135,
        "mon": monitor_number,
    }
    output = "sct-mon{mon}_{top}x{left}_{width}x{height}.png".format(**monitor)

    # Grab the data
    img = sct.grab(monitor)

    # Save to the picture file
    # mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
    # print(output)
    cv2.imshow('test', np.array(img))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

for window in current_windows():
    name = win_name(window)
    size = list(win_size(window))
    print(f'{name}: {size}')