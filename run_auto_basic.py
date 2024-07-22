"""runs the check screenshot and flick every n seconds"""
import time
import cv2
from monst import MonstDatabase, MonstBattleState
from adb_utils import ADBUtils

POLL_DELAY = 5
DEBUG_MODE = False

db = MonstDatabase("config")

device = ADBUtils.from_device()

input("enter any key to start: ")

#battle should finish in 60*20 = 20 minutes
for i in range(1200):
    battle_img = device.take_screenshot_numpy()
    battle_state = MonstBattleState(battle_img=battle_img, monst_db=db)

    found_players = list(battle_state.player_coords.keys())
    active_player = battle_state.active_player_num
    found_other_players = [player for player in found_players if player != active_player] if found_players else []

    if active_player in found_players and found_other_players:
        pt1, pt2 = MonstBattleState._calculate_swipe_coords(battle_state.player_coords[active_player], battle_state.player_coords[found_other_players[0]])
    else:
        print("default to 45 degree")
        pt1, pt2 = MonstBattleState._calculate_swipe_coords((0, 0), (1, 1))

    #pt2 to pt1 swipe motion will result in arrow from pt1 to pt2
    device.swipe(pt2, pt1)

    if DEBUG_MODE:
        cv2.imwrite(f"test_run_imgs/{i}.png", battle_img)
        with open(f"test_run_imgs/{i}.txt", "w") as file:
            file.write("all player positions:\n")
            file.write(str(battle_state.player_coords) + "\n")
            file.write(f"active player: {active_player}")


    time.sleep(POLL_DELAY)