import random

from euchre.round import Round
from state_encoding.minimal_rl import encode_state
from euchre.players import PLAYER_COUNT

player = random.randint(0, PLAYER_COUNT - 1)

def print_random_encoded_game_state():
    round = Round()

    action_count = random.randint(0, 16)

    i = 0
    while True:
        if i >= action_count and round.current_player == player:
            print(encode_state(round))
            return

        actions = round.get_actions()
        action = random.choice(list(actions))
        round.take_action(action)

        if round.finished:
            break

        i += 1
    
    print_random_encoded_game_state()

print_random_encoded_game_state()