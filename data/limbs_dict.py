# limbs connection dict

CONNECTION_TORSO_TOP = (11, 12)
CONNECTION_TORSO_LEFT = (11, 23)
CONNECTION_TORSO_RIGHT = (12, 24)
CONNECTION_TORSO_BOTTOM = (23, 24)
CONNECTION_ARM_LEFT = (11, 13)
CONNECTION_ARM_RIGHT = (12, 14)
CONNECTION_FOREARM_LEFT = (13, 15)
CONNECTION_FOREARM_RIGHT = (14, 16)
CONNECTION_UPPER_LEG_LEFT = (23, 25)
CONNECTION_UPPER_LEG_RIGHT = (24, 26)
CONNECTION_LOWER_LEG_LEFT = (25, 27)
CONNECTION_LOWER_LEG_RIGHT = (26, 28)

limbs_connect_dict_description = {
    (11, 12): "CONNECTION_TORSO_TOP",
    (11, 23): "CONNECTION_TORSO_LEFT",
    (12, 24): "CONNECTION_TORSO_RIGHT",
    (23, 24): "CONNECTION_TORSO_BOTTOM",
    (11, 13): "CONNECTION_ARM_LEFT",
    (12, 14): "CONNECTION_ARM_RIGHT",
    (13, 15): "CONNECTION_FOREARM_LEFT",
    (14, 16): "CONNECTION_FOREARM_RIGHT",
    (23, 25): "CONNECTION_UPPER_LEG_LEFT",
    (24, 26): "CONNECTION_UPPER_LEG_RIGHT",
    (25, 27): "CONNECTION_LOWER_LEG_LEFT",
    (26, 28): "CONNECTION_LOWER_LEG_RIGHT",
}

limbs_connect_dict_arr = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (12, 14), (13, 15), (14, 16),
                                    (23, 25), (24, 26), (25, 27), (26, 28)])
# limb connect dict by index
TORSO_TOP = 0
TORSO_LEFT = 1
TORSO_RIGHT = 2
TORSO_BOTTOM = 3
ARM_LEFT = 4
ARM_RIGHT = 5
FOREARM_LEFT = 6
FOREARM_RIGHT = 7
UPPER_LEG_LEFT = 8
UPPER_LEG_RIGHT = 9
LOWER_LEG_LEFT = 10
LOWER_LEG_RIGHT = 11

limbs_connect_dict_arr_index_description = {
    0: 'TORSO_TOP',
    1: 'TORSO_LEFT',
    2: 'TORSO_RIGHT',
    3: 'TORSO_BOTTOM',
    4: 'ARM_LEFT',
    5: 'ARM_RIGHT',
    6: 'FOREARM_LEFT',
    7: 'FOREARM_RIGHT',
    8: 'UPPER_LEG_LEFT',
    9: 'UPPER_LEG_RIGHT',
    10: 'LOWER_LEG_LEFT',
    11: 'LOWER_LEG_RIGHT',
}


def get_limb_description_by_limb(limb: int):
    return limbs_connect_dict_arr_index_description[limb]
