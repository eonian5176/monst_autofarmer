import os
from pathlib import Path

from adb_utils import ADBUtils
from utils import template_match, resize

import cv2
import pandas as pd

device = ADBUtils.from_device()
device.take_screenshot(Path(os.getcwd()), "monst_battle")

unit = "chihaya"

unit_centers = pd.read_csv("ball art/centers.csv", dtype={"unit": "string", "x": "int32", "y": "int32"}).set_index("unit")

# device.swipe_relative((0.3, 0.3), (0.5, 0.5))
start_pt = template_match(image_path="test case/in bubble.png", template_path=f"ball art/{unit}.png", threshold=0.7)
center_offset_data = unit_centers.loc[unit]
unit_coords = (start_pt[1] + center_offset_data["x"], start_pt[0] + center_offset_data["y"])
print(unit_coords)
img = cv2.imread("monst_battle.png")
img = cv2.circle(img, unit_coords, 10, (255, 0, 0), thickness=-1)
img = resize(img, 0.5)
cv2.imshow("a", img)
cv2.waitKey(-1)