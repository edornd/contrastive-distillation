ICL_ISPRS = {
    # full segmentation, without incremental stuff
    "offline": {
        0: [0, 1, 2, 3, 4, 5]
    },
    # need to increment every index by 1 in the dataset, since the 'background' class does not really exist
    # the 0 class is surface, which could be used as background, but here doesn't make sense
    "222a": {
        0: [0, 2],    # surface (0), low vegetation (2)
        1: [1, 3],    # building (1), high vegetation(3)
        2: [4, 5]    # car (4), clutter (5)
    },
    "321": {
        0: [1, 3, 5],    # building(1), high veg. (3), clutter (5)
        1: [0, 2],    # surface (0), low veg. (2)
        2: [4]    # car (4)
    },
    "6s": {
        0: [1],
        1: [3],
        2: [0],
        3: [2],
        4: [4],
        5: [5]
    }
}
