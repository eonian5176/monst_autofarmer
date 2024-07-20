import sys
import os
import time

import cv2
from pathlib import Path

from adb_utils import ADBUtils

screenshot_name = sys.argv[1]
device = ADBUtils(screen_width=1080, screen_height=2340)
print(device.screen_width, device.screen_height)

st = time.time()
# device.take_screenshot(Path(os.getcwd()), screenshot_name=screenshot_name, screenshot_format="png")
np_array = device.take_screenshot_numpy()
print(np_array.shape)
ed = time.time()

cv2.imshow("a", np_array)
cv2.waitKey(-1)

print(ed-st)