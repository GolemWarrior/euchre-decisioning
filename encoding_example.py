import random

from euchre.round import Round
from state_encoding.minimal_rl import encode_state as minimal_encode_state
from state_encoding.minimal_rl import encode_state_with_card_prob as minimal_encode_state_with_card_prob
from state_encoding.verbose_rl import encode_state as verbose_encode_state
from euchre.players import PLAYER_COUNT

player = random.randint(0, PLAYER_COUNT - 1)

def print_random_encoded_game_state():
    round = Round()

    action_count = random.randint(0, 24)

    i = 0
    while True:
        if i >= action_count and round.current_player == player:
            print("Minimal:")
            print(minimal_encode_state(round))

            print("\nMinimal with card probabilities:")
            print(minimal_encode_state_with_card_prob(round))

            print("\nVerbose:")
            print(verbose_encode_state(round))

            return

        actions = round.get_actions()
        action = random.choice(list(actions))
        round.take_action(action)

        if round.finished:
            break

        i += 1
    
    print_random_encoded_game_state()

print_random_encoded_game_state()