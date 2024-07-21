from monst import MonstDatabase, MonstBattleState
import cv2
from adb_utils import ADBUtils


db = MonstDatabase("config")

adb_interfacer = ADBUtils.from_device()

battle_img = adb_interfacer.take_screenshot_numpy()
battle_state = MonstBattleState(battle_img=battle_img, monst_db=db)

print(battle_state.active_player_num)