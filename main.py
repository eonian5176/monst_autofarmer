import os
from pathlib import Path

from adb_utils import ADBUtils
from utils import template_match, resize

import cv2
import pandas as pd
import sys


unit = sys.argv[1]
picture = "test case/battle images/label test easy.png"

unit_centers = pd.read_csv("unit number art/offsets.csv", dtype={"icon": "string", "x": "int32", "y": "int32"}).set_index("icon")

# device.swipe_relative((0.3, 0.3), (0.5, 0.5))
start_pt = template_match(image_path=picture, template_path=f"unit number art/{unit}.png", grayscale_match=False, threshold=0.75)
center_offset_data = unit_centers.loc[unit]
unit_coords = (start_pt[0] + center_offset_data["x"], start_pt[1] + center_offset_data["y"])
print(unit_coords)
img = cv2.imread(picture)
img = cv2.circle(img, unit_coords, 20, (255, 0, 0), thickness=-1)
img = resize(img, 0.5)
cv2.imshow("a", img)
cv2.waitKey(-1)