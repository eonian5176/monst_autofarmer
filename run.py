from monst import MonstDatabase, MonstBattleState
import cv2
from adb_utils import ADBUtils
from utils import resize


db = MonstDatabase("config")

device = ADBUtils.from_device()

for i in range(12):
    battle_img = device.take_screenshot_numpy()
    battle_state = MonstBattleState(battle_img=battle_img, monst_db=db)
    
    active_player = battle_state.active_player_num
    next_player = active_player % 4 + 1

    print("active player", active_player)
    player_position = battle_state.player_coords.get(active_player)
    next_player_position = battle_state.player_coords.get(next_player)
    print("player and next player positions:", player_position, next_player_position)
    if player_position is not None and next_player_position is not None:
        pt1, pt2 = MonstBattleState._calculate_swipe_coords(battle_state.player_coords[active_player], battle_state.player_coords[next_player])
    else:
        print("default to 45 degree")
        pt1, pt2 = MonstBattleState._calculate_swipe_coords((0, 0), (1, 1))

    #pt2 to pt1 swipe motion will result in arrow from pt1 to pt2
    device.swipe(pt2, pt1)

    input("enter any key to continue: ")