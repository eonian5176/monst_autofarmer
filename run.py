from monst import MonstDatabase, MonstBattleState
from adb_utils import ADBUtils

db = MonstDatabase("config")

device = ADBUtils.from_device()

input("enter any key to start:")

for i in range(100):
    battle_img = device.take_screenshot_numpy()
    battle_state = MonstBattleState(battle_img=battle_img, monst_db=db)
    
    active_player = battle_state.active_player_num
    # next_player = active_player % 4 + 1

    print("active player", active_player)
    print("all player positions:", battle_state.player_coords)

    found_players = list(battle_state.player_coords.keys())
    found_other_players = [player for player in found_players if player != active_player] if found_players else []

    if active_player in found_players and found_other_players:
        pt1, pt2 = MonstBattleState._calculate_swipe_coords(battle_state.player_coords[active_player], battle_state.player_coords[found_other_players[0]])
    else:
        print("default to 45 degree")
        pt1, pt2 = MonstBattleState._calculate_swipe_coords((0, 0), (1, 1))

    #pt2 to pt1 swipe motion will result in arrow from pt1 to pt2
    device.swipe(pt2, pt1)

    input("enter any key to continue: ")