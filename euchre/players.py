from enum import IntEnum

PLAYERS = ["Player0_Team0", "Player1_Team1", "Player2_Team0", "Player3_Team1"]  # Order is important
PLAYER_COUNT = len(PLAYERS)
EPlayer = IntEnum('EPlayers', [(player, index) for index, player in enumerate(PLAYERS)])

def get_teammate(eplayer):
    # Gets the other player on the same team.
    # (Assumes 4 players and same team players are next to each other)
    return EPlayer((eplayer + 2) % 4)

def are_eplayers_same_team(eplayer1, eplayer2):
    return get_teammate(eplayer1) == eplayer2

def eplayer_to_team_index(eplayer):
    return eplayer % 2

def get_other_team_index(team_index):
    assert team_index == 0 or team_index == 1, "Team index is either 0 or 1!"
    return 1 - team_index  # {0: 1, 1: 0}

def get_clockwise_player(eplayer):
    return EPlayer((eplayer + 1) % 4)