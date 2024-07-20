import os
import subprocess

from utils import Point
from typing import Iterable
from pathlib import Path

class ADBUtils:
    screen_width: int
    screen_height: int

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def validate_coord(self, coord: Point) -> Point:
        """
        given a coord, check if it is valid for the device specification, return
        coord if valid, else throw error
        """
        if not (0 <= coord.x <= self.screen_width and 0 <= coord.y <= self.screen_height):
            raise ValueError(f"({coord[0]}, {coord[1]}) out of screen scope ({self.screen_width},{self.screen_height})")
        
        return coord
        
    def validate_coords(self, coords: Iterable[Point]) -> Iterable[Point]:
        """
        return the iterable of coords if all valid, else throw error
        """
        for coord in coords:
            self.validate_coord(coord)
        
        return coords

    @classmethod
    def from_device(cls) -> "ADBUtils":
        """
        calls an adb process that determines resolution of device and assigns it
        to member variables
        """
        result = subprocess.run(["adb", "shell", "wm", "size"], capture_output=True)

        result.check_returncode()

        x, y = map(int, result.stdout.decode(encoding="utf-8").split(":")[1].strip().split("x"))

        return cls(x, y)

    @staticmethod
    def take_screenshot(save_path: Path, screenshot_name: str, screenshot_format: str = "png") -> None:
        """
        calls an adb process to take screenshot and save it to a given path with
        a specified screenshot format, default png
        """
        write_path = os.path.join(save_path, f"{screenshot_name}.{screenshot_format}")
        result = subprocess.run(["adb", "exec-out", "screencap", "-p"], capture_output=True)

        result.check_returncode()

        with open(write_path, "wb") as file:
            file.write(result.stdout)

    def pct_to_point(self, pct_x: float, pct_y: float) -> Point:
        """
        convert percentages on screen to valid integer pixel values (round to nearest)
        """
        return self.validate_coord(Point(int(pct_x*self.screen_width), int(pct_y*self.screen_height)))

    def tap(self, coord: Point) -> None:
        """
        tap exact coord relative to screen
        """
        self.validate_coord(coord)
        subprocess.run(["adb", "shell", "input", "tap", str(coord[0]), str(coord[1])])

    def tap_relative(self, pct_x: float, pct_y: float) -> None:
        """
        tap relative coord on screen
        """
        self.tap(self.pct_to_point(pct_x, pct_y))

    def swipe(self, start: Point, end: Point) -> None:
        """
        swipe exact coords on screen
        """
        self.validate_coords([start, end])
        subprocess.run(["adb", "shell", "input", "swipe", str(start[0]), str(start[1]), str(end[0]), str(end[1])])

    def swipe_relative(self, start_pcts: tuple[float, float], end_pcts: tuple[float, float]) -> None:
        """
        swipe relative coords on screen
        """
        self.swipe(self.pct_to_point(start_pcts[0], start_pcts[1]), self.pct_to_point(end_pcts[0], end_pcts[1]))



        

