# 0:  background
# 1:  ship
# 2:  storage_tank
# 3:  baseball_diamond
# 4:  tennis_court
# 5:  basketball_court
# 6:  ground_track_field
# 7:  bridge
# 8:  large_vehicle
# 9:  small_vehicle
# 10: helicopter
# 11: swimming_pool
# 12: roundabout
# 13: soccer_ball_field
# 14: plane
# 15: harbor
ICL_ISAID = {
    "offline": {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    },
    "4s": {
        0: [0, 3, 4, 5, 6, 11, 13],    # sports fields
        1: [7, 8, 9, 12],    # traffic and vehicles
        2: [1, 2, 15],    # ships, harbors, storage tank
        3: [10, 14]    # helicopters and planes
    }
}
