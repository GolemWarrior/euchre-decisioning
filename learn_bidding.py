import numpy as np
import random

from euchre.players import eplayer_to_team_index, get_other_team_index
from euchre.deck import ESuit, get_ecard_esuit, get_ecard_erank, get_same_color_esuit, SUITS, RANKS, JACK
from euchre.round import Round, suit_choice_action_to_esuit_map, PLAY_CARD_ACTIONS, FIRST_BIDDING_STATE, SECOND_BIDDING_STATE, CHOOSING_ESUIT_STATE, DEALER_DISCARD_STATE, DECIDING_GOING_ALONE_STATE, GO_ALONE, DONT_GO_ALONE, ORDER_UP, PASS

MAX_RANK = len(RANKS)

esuit_to_suit_choice_action = {esuit: choice_action for choice_action, esuit in suit_choice_action_to_esuit_map.items()}

def encode_hand(hand, trump_esuit):
    trump_count = 0
    max_trump = 0
    trump_sum = 0
    for ecard in hand:
        if get_ecard_esuit(ecard) == trump_esuit:
            trump_count += 1
            max_trump = max(max_trump, int(get_ecard_erank(ecard)))
            trump_sum += int(get_ecard_erank(ecard)) 

    trump_count = max(trump_count, 1)  # Avoid dividing by zero
    avg_trump = trump_sum / trump_count

    has_right_bower = 0
    has_left_bower = 0
    for ecard in hand:
        if get_ecard_erank(ecard) == JACK:
            if get_ecard_esuit(ecard) == trump_esuit:
                has_right_bower = 1
            elif get_ecard_esuit(ecard) == get_same_color_esuit(trump_esuit):
                has_left_bower = 1

    non_trump_count = 0
    non_trump_max = 0
    non_trump_sum = 0

    for ecard in hand:
        if get_ecard_esuit(ecard) != trump_esuit:
            non_trump_count += 1
            non_trump_max = max(non_trump_max, int(get_ecard_erank(ecard)))
            non_trump_sum += int(get_ecard_erank(ecard))

    non_trump_count = max(non_trump_count, 1)  # Avoid dividing by zero
    non_trump_avg = non_trump_sum / non_trump_count

    dont_have_suits = np.ones(len(SUITS))
    for ecard in hand:
        esuit = get_ecard_esuit(ecard)
        dont_have_suits[int(esuit)] = 0

    void_suits = np.sum(dont_have_suits)

    return np.array([
        trump_count,
        max_trump,
        avg_trump,
        has_right_bower,
        has_left_bower,
        non_trump_count,
        non_trump_max,
        non_trump_avg,
        void_suits
    ])

HAND_ENCODING_LENGTH = len(encode_hand([0] * len(SUITS), 0))

def learn_from_hands(hands, trump_esuits, scores):
    # Takes in a list of maker hands, a list of their selected trump suits, and a list of their scores (IMPORTANT: with opponent points counting as negative!)

    sample_count = len(hands)

    x_with_bias = np.ones(shape=(sample_count, HAND_ENCODING_LENGTH+1))
    for i, hand, trump_esuit in zip(range(sample_count), hands, trump_esuits):
        hand_encoding = encode_hand(hand, trump_esuit)
        x_with_bias[i,0:HAND_ENCODING_LENGTH] = hand_encoding

    y = np.array(scores)

    theta = np.linalg.inv(x_with_bias.T @ x_with_bias) @ x_with_bias.T @ y

    weights = theta[1:]
    bias = theta[0]

    return weights, bias

def predict_expected_score(hand, trump_esuit, weights, bias):
    hand_encoding = encode_hand(hand, trump_esuit)
    predicted_expected_score = bias + weights @ hand_encoding

    return predicted_expected_score

BIDDING_ESTATES = [FIRST_BIDDING_STATE, SECOND_BIDDING_STATE, CHOOSING_ESUIT_STATE, DEALER_DISCARD_STATE, DECIDING_GOING_ALONE_STATE]
def get_best_bidding_action(round, order_weights, order_bias, order_alone_weights, order_alone_bias):
    assert round.estate in BIDDING_ESTATES, "Can only get best bidding action in bidding states!"

    eplayer = round.get_current_player()
    hand = round.hands[int(eplayer)]

    if round.estate == FIRST_BIDDING_STATE:
        trump_esuit = get_ecard_esuit(round.upcard)
        expected_order_score = max(
            predict_expected_score(hand, trump_esuit, order_weights, order_bias),
            predict_expected_score(hand, trump_esuit, order_alone_weights, order_alone_bias)
            )

        if expected_order_score > 0:
            return ORDER_UP
        else:
            return PASS

    elif round.estate == DEALER_DISCARD_STATE:
        # Pick the weakest card based on: is_bower, is_trump, rank, -same_suit_count
        ranking_tuples = []
        for ecard in hand:
            is_bower = 0
            if get_ecard_erank(ecard) == JACK and (get_ecard_esuit(ecard) == round.trump_esuit or get_ecard_esuit(ecard) == get_same_color_esuit(round.trump_esuit)):
                is_bower = True

            is_trump = int(get_ecard_esuit(ecard) == round.trump_esuit)

            rank = int(get_ecard_erank(ecard))

            same_suit_count = 0
            for other_ecard in hand:
                if other_ecard == ecard:
                    continue

                if get_ecard_esuit(other_ecard) == get_ecard_esuit(ecard):
                    same_suit_count += 1

            ranking_tuples.append((is_bower, is_trump, rank, -same_suit_count))

        min_ecard_index = ranking_tuples.index(min(ranking_tuples))

        return PLAY_CARD_ACTIONS[min_ecard_index]

    elif round.estate == SECOND_BIDDING_STATE or round.estate == CHOOSING_ESUIT_STATE:
        best_trump_esuit = ESuit(0)
        best_expected_score = -float("inf")
        for i in range(len(SUITS)):
            potential_trump_esuit = ESuit(i)
            expected_score = max(
                predict_expected_score(hand, potential_trump_esuit, order_weights, order_bias),
                predict_expected_score(hand, potential_trump_esuit, order_alone_weights, order_alone_bias)
                )
            if expected_score > best_expected_score:
                best_expected_score = expected_score
                best_trump_esuit = potential_trump_esuit

        if round.estate == SECOND_BIDDING_STATE:
            if best_expected_score > 0:
                return ORDER_UP
            else:
                return PASS
        elif round.estate == CHOOSING_ESUIT_STATE:
            return esuit_to_suit_choice_action[best_trump_esuit]

    elif round.estate == DECIDING_GOING_ALONE_STATE:
        trump_esuit = round.trump_esuit
        dont_go_alone = predict_expected_score(hand, trump_esuit, order_weights, order_bias)
        go_alone = predict_expected_score(hand, trump_esuit, order_alone_weights, order_alone_bias)

        if go_alone > dont_go_alone:
            return GO_ALONE
        else:
            return DONT_GO_ALONE

SAMPLE_COUNT = 100000
def rough_learn_bidding(sample_count=SAMPLE_COUNT):
    data = {
        GO_ALONE: {
            "hands": [],
            "trumps": [],
            "scores": []
        },
        DONT_GO_ALONE: {
            "hands": [],
            "trumps": [],
            "scores": []
        }
    }

    print("Generating data ...")

    for deciding_alone_action in data.keys():
        for i in range(sample_count):
            round = Round()

            while not round.finished and round.estate != DECIDING_GOING_ALONE_STATE:
                round.take_action(random.choice(list(round.get_actions())))

            if round.finished:
                continue

            round.take_action(deciding_alone_action)
            hand = round.hands[int(round.maker)].copy()

            while not round.finished:
                round.take_action(random.choice(list(round.get_actions())))

            maker = round.maker
            maker_team = eplayer_to_team_index(maker)
            other_team = get_other_team_index(maker_team)

            trump_esuit = round.trump_esuit
            score = round.round_points[maker_team] - round.round_points[other_team]

            data[deciding_alone_action]["hands"].append(hand)
            data[deciding_alone_action]["trumps"].append(trump_esuit)
            data[deciding_alone_action]["scores"].append(score)

            if i % 10000 == 0:
                print(f"Ran simulation {i} for {deciding_alone_action.name}!")

    print("Generated data!")
    print("Learning ...")

    together_weights, together_biases = learn_from_hands(data[DONT_GO_ALONE]["hands"], data[DONT_GO_ALONE]["trumps"], data[DONT_GO_ALONE]["scores"])
    alone_weights, alone_biases = learn_from_hands(data[GO_ALONE]["hands"], data[GO_ALONE]["trumps"], data[GO_ALONE]["scores"])

    print("\nFinished learning!")
    print("together_weights:")
    print(together_weights)
    print("together_biases:")
    print(together_biases)
    print("alone_weights:")
    print(alone_weights)
    print("alone_biases:")
    print(alone_biases)

    return together_weights, together_biases, alone_weights, alone_biases

if __name__ == "__main__":
    together_weights, together_biases, alone_weights, alone_biases = rough_learn_bidding()

    TEST_ROUNDS = 100000

    total_won_points = 0
    total_lost_points = 0
    total_wins = 0
    total_losses = 0

    print("Running test rounds ...")

    for i in range(TEST_ROUNDS):
        round = Round()

        while not round.finished:
            if round.estate in BIDDING_ESTATES and eplayer_to_team_index(round.get_current_player()) == 0:
                pred_best_action = get_best_bidding_action(round, together_weights, together_biases, alone_weights, alone_biases)
                round.take_action(pred_best_action)
            else:
                round.take_action(random.choice(list(round.get_actions())))

        total_won_points += round.round_points[0]
        total_lost_points += round.round_points[1]
        total_wins += int(round.round_points[0] > round.round_points[1])
        total_losses += int(round.round_points[0] < round.round_points[1])

    total_points = total_won_points - total_lost_points

    print("Finished test rounds!")
    print(f"Average win points minus loss points: {total_points / (total_wins + total_losses):.2f} points")
    print(f"Average points: {total_won_points / (total_wins + total_losses):.2f} points")
    print(f"Win percentage: {total_wins / (total_wins + total_losses) * 100:.2f}%")