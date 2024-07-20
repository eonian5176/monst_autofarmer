from typing import NamedTuple, Optional
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

Point = tuple[int, int]

def resize(img: npt.NDArray[np.uint8], scale: float, method: int=cv2.INTER_AREA) -> npt.NDArray[np.uint8]:
    """convenient wrapper"""
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    # resize image
    return cv2.resize(img, dim, interpolation = method)

def template_match(image_path: Path, template_path: Path, grayscale_match:bool=True, threshold: float=0.5) -> Optional[Point]:
    """
    finds small template (image) in larger image using cross correlation
    """
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    if grayscale_match:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    print(result.max())
    print(result.shape)
    if np.max(result) < threshold:
        return

    match_start_point = np.unravel_index(np.argmax(result), shape = result.shape)

    assert len(match_start_point) == 2

    return match_start_point


