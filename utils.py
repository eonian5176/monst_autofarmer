from typing import NamedTuple, Optional
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
import math
from matplotlib import pyplot as plt
from typing import Any

Point = tuple[int, int]
ImageObject = npt.NDArray[np.uint8]

class TemplateMatchError(Exception):
    pass

def resize(img: npt.NDArray[np.uint8], scale: float, method: int=cv2.INTER_AREA) -> npt.NDArray[np.uint8]:
    """convenient wrapper"""
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    # resize image
    return cv2.resize(img, dim, interpolation = method)

def template_match(
    image: ImageObject, 
    template: ImageObject, 
    method: int=cv2.TM_CCOEFF_NORMED, 
    grayscale_match:bool=False, 
    threshold: Optional[float]=None
) -> tuple[int, int, int]:
    """
    wrapper around template match with grayscale conversion feature and returning
    first position of max.

    if threshold used, throw TemplateMatchError if no match found

    returns:
    x: x-coord of first max corr
    y: y-coord of first max corr
    corr: max correlation value
    """
    if grayscale_match:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        img = image
        tmpl = template

    result = cv2.matchTemplate(img, tmpl, method)

    max_corr = np.max(result)
    if threshold is not None and max_corr < threshold:
        raise TemplateMatchError(f"template not found using threshold {threshold}")

    match_start_point = np.unravel_index(np.argmax(result), shape = result.shape)

    assert len(match_start_point) == 2

    return (match_start_point[1], match_start_point[0], max_corr)

def np_rmse(arr1: npt.NDArray[Any], arr2: npt.NDArray[Any]) -> np.float64:
    if arr1.shape != arr2.shape:
        raise ValueError("rmse arrays shape not identical")
    
    return np.sqrt(np.mean((arr2-arr1)**2))
