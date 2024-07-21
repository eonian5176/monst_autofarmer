import pandas as pd
import numpy as np
import numpy.typing as npt
import os
import cv2
from utils import Point, ImageObject, np_rmse
from pathlib import Path
from functools import cached_property

class MonstDatabase:
    ball_arts: dict[str, ImageObject]
    ball_centers: dict[str, Point]

    turn_icon_arts: dict[int, ImageObject]
    icon_to_ball_offsets: dict[int, Point]

    turn_border_imgs: dict[int, ImageObject]
    turn_border_badged_imgs: dict[int, ImageObject]
    turn_border_positions: dict[int, tuple[Point, Point]]

    @staticmethod
    def fetch_ball_arts(ball_arts_path: Path) -> dict[str, ImageObject]:
        ball_arts = {}
        
        for file in os.listdir(ball_arts_path):
            if file.endswith(".png"):
                ball_arts[file.split(".")[0]] = cv2.imread(os.path.join(ball_arts_path, file))
        
        return ball_arts
    
    @staticmethod
    def fetch_ball_centers(ball_arts_path: Path) -> dict[str, Point]:
        ball_centers = {}
        with open(os.path.join(ball_arts_path, "centers.csv"), "r") as centers_file:
            centers_file.readline()
            for line in centers_file:
                vals = line.split(",")
                ball_centers[vals[0]] = (int(vals[1]), int(vals[2]))
        
        return ball_centers
    
    @staticmethod
    def fetch_turn_icons(turn_icons_path: Path) -> dict[str, ImageObject]:
        turn_icon_arts = {}
        for i in range(1, 5):
            turn_icon_arts[i] = cv2.imread(os.path.join(turn_icons_path, f"{i}p.png"))

        return turn_icon_arts

    @staticmethod
    def fetch_icon_to_ball_offsets(turn_icons_path: Path) -> dict[str, Point]:
        icon_to_ball_offsets = {}
        with open(os.path.join(turn_icons_path, "offsets.csv"), "r") as offsets_file:
            offsets_file.readline()
            for line in offsets_file:
                vals = line.split(",")
                icon_to_ball_offsets[int(vals[0])] = (int(vals[1]), int(vals[2]))
        
        return icon_to_ball_offsets        
    
    @staticmethod
    def fetch_turn_border_imgs(turn_borders_path: Path, badged: bool) -> dict[str, ImageObject]:
        turn_border_imgs = {}

        for i in range(1, 5):
            filename = str(i) + ("b" if badged else "")
            turn_border_imgs[i] = cv2.imread(os.path.join(turn_borders_path, f"{filename}.png"))

        return turn_border_imgs
    
    @staticmethod
    def fetch_turn_border_positions(turn_borders_path: Path) -> dict[str, tuple[Point, Point]]:
        turn_border_positions = {}
        with open(os.path.join(turn_borders_path, "frame_positions.csv"), "r") as offsets_file:
            offsets_file.readline()
            for line in offsets_file:
                vals = line.split(",")
                turn_border_positions[int(vals[0])] = (
                    (int(vals[1]), int(vals[2])), 
                    (int(vals[3]), int(vals[4])),
                )
        
        return turn_border_positions   


    def __init__(self, config_path: Path):
        """
        What files you need to ensure to have in config folder:
            -ball art folder
                -ball arts pngs
                -centers.csv
            -turn icon folder
                -turn icon pngs
                -offsets.csv
            -turn border folder
                -turn border pngs
                -frame_positions.csv
        """
        ball_arts_path = os.path.join(config_path, "ball art")
        turn_icons_path = os.path.join(config_path, "turn icon")
        turn_borders_path = os.path.join(config_path, "turn border")

        self.ball_arts = MonstDatabase.fetch_ball_arts(ball_arts_path)
        self.ball_centers = MonstDatabase.fetch_ball_centers(ball_arts_path)

        self.turn_icon_arts = MonstDatabase.fetch_turn_icons(turn_icons_path)
        self.turn_icon_offsets = MonstDatabase.fetch_icon_to_ball_offsets(turn_icons_path)

        self.turn_border_imgs = MonstDatabase.fetch_turn_border_imgs(turn_borders_path, badged=False)
        self.turn_border_badged_imgs = MonstDatabase.fetch_turn_border_imgs(turn_borders_path, badged=True)
        self.turn_border_positions = MonstDatabase.fetch_turn_border_positions(turn_borders_path)


class MonstBattleState:
    battle_img: ImageObject
    monst_db: MonstDatabase

    def __init__(self, battle_img: ImageObject, monst_db: MonstDatabase):
        self.battle_img = battle_img
        self.monst_db = monst_db

    @cached_property
    def turn_borders(self) -> dict[int, ImageObject]:
        """extracts the 4 portions of battle_img corresponding to turn borders"""
        borders = {}
        for i in range(1, 5):
            start, end = self.monst_db.turn_border_positions[i]
            borders[i] = self.battle_img[start[1]:end[1], start[0]:end[0]]
        
        return borders

    @cached_property
    def active_player_num(self) -> int:
        """
        looks at battle_img and compares it to monst_db.turn_border_imgs and
        monst_db.turn_border_badged_imgs to find the most likely candidate
        active player (who is supposed to move in the img)
        """
        normal_diff = [
            np_rmse(
                self.turn_borders[i], 
                self.monst_db.turn_border_imgs[i], 
            ) for i in range(1, 5)
        ]
        
        badged_diff = [
            np_rmse(
                self.turn_borders[i], 
                self.monst_db.turn_border_badged_imgs[i], 
            ) for i in range(1, 5)
        ]


        player_diff = np.array([normal_diff, badged_diff], dtype=np.float64).min(axis=0)
        print(player_diff)
        return player_diff.argmax()+1







    