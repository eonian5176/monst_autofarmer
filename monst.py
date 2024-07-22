import pandas as pd
import numpy as np
import numpy.typing as npt
import os
import cv2
from utils import Point, ImageObject, np_rmse, template_match, TemplateMatchError
from pathlib import Path
from functools import cached_property
from skimage.metrics import structural_similarity as ssim
from typing import Optional
import math

class MonstDatabase:
    ball_arts: dict[str, ImageObject]
    ball_centers: dict[str, Point]

    turn_icon_arts: dict[int, list[ImageObject]]
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
    def fetch_turn_icons(turn_icons_path: Path) -> dict[str, list[ImageObject]]:
        """
        1p has 1 art- red
        2p has 2 art- red, yellow
        3p has 3 art- red, yellow, green
        3p has 4 art- red, yellow, green, blue
        """
        turn_icon_arts: dict[int, list[ImageObject]] = {}
        for i in range(1, 5):
            turn_icon_arts[i] = []

            for suffix in ["", "_y", "_b", "_g"]:
                art_full_path = os.path.join(turn_icons_path, f"{i}p" + suffix + ".png")
                if os.path.exists(art_full_path):
                    turn_icon_arts[i].append(cv2.imread(art_full_path))

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

    SWIPE_CIRCLE_CENTER = (540, 920)
    SWIPE_CIRCLE_RADIUS = 500

    def __init__(self, battle_img: ImageObject, monst_db: MonstDatabase):
        self.battle_img = battle_img
        self.monst_db = monst_db

    @staticmethod
    def _calculate_angle(from_pt: Point, to_pt: Point):
        """
        returns angle in radians of arrow from from_pt to to_pt
        """
        delta_x = to_pt[0] - from_pt[0]
        delta_y = to_pt[1] - from_pt[1]
        angle = math.atan2(delta_y, delta_x)
        return angle
    
    @staticmethod
    def _calculate_swipe_coords(from_pt: Point, to_pt: Point) -> tuple[Point, Point]:
        """
        angle refers to the direction from from_pt to to_pt, so adding angular
        projection attains endpoint (2), and subtracting attains start (1)
        """
        angle = MonstBattleState._calculate_angle(from_pt, to_pt)

        x_diff = MonstBattleState.SWIPE_CIRCLE_RADIUS*math.cos(angle)
        y_diff =MonstBattleState.SWIPE_CIRCLE_RADIUS*math.sin(angle)

        x2 = round(MonstBattleState.SWIPE_CIRCLE_CENTER[0] + x_diff)
        y2 = round(MonstBattleState.SWIPE_CIRCLE_CENTER[1] + y_diff)

        x1 = round(MonstBattleState.SWIPE_CIRCLE_CENTER[0] - x_diff)
        y1 = round(MonstBattleState.SWIPE_CIRCLE_CENTER[1] - y_diff)

        return ((x1, y1), (x2, y2))

    @cached_property
    def _turn_borders(self) -> dict[int, ImageObject]:
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

        note:
        ssim seem to work much better than rmse for now

        ssim works so well that it should be able to tell if no bumps exist, then it
        means no player turn right now, which we can skip the motion
        """
        normal_diff = [
            ssim(
                self._turn_borders[i], 
                self.monst_db.turn_border_imgs[i], 
                channel_axis=2
            ) for i in range(1, 5)
        ]
        
        badged_diff = [
            ssim(
                self._turn_borders[i], 
                self.monst_db.turn_border_badged_imgs[i],
                channel_axis=2
            ) for i in range(1, 5)
        ]

        player_diff = np.array([normal_diff, badged_diff], dtype=np.float64).max(axis=0)
        print(player_diff)
        return player_diff.argmin()+1

    @cached_property
    def player_coords(self) -> dict[int, Point]:
        """
        return dict of player ball locations for player turn icons template match can
        find using a .8 threshold

        if none of the templates for a player's icon can match battle img above 
        threshold, that player will not exist in dict
        """
        player_coords = {}

        for player in range(1,5):
            coords = []
            coeffs = []
            for img in self.monst_db.turn_icon_arts[player]:
                try:
                    x, y, coeff = template_match(self.battle_img, img, threshold=0.8)
                    coords.append((x,y))
                    coeffs.append(coeff)
                except TemplateMatchError:
                    pass

            if coords:
                max_coeff_idx = np.array(coeffs, np.float64).argmax()
                turn_icon_x, turn_icon_y = coords[max_coeff_idx]
                player_coords[player] = (turn_icon_x + self.monst_db.turn_icon_offsets[player][0], turn_icon_y + self.monst_db.turn_icon_offsets[player][1])

        return player_coords
    
    

    


        





    