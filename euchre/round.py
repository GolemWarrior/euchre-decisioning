from enum import IntEnum
import random
import numpy as np

from .deck import ESuit, ECard, Deck, get_ecard_esuit, get_ecard_erank, get_same_color_esuit, get_ecard_score, JACK
from .players import EPlayer, get_clockwise_player, eplayer_to_team_index, get_other_team_index, get_teammate

ROUND_STATES = ["FIRST_BIDDING", "DEALER_DISCARD", "DECIDING_GOING_ALONE", "SECOND_BIDDING", "CHOOSING_ESUIT", "PLAYING"]
RoundEState = IntEnum("RoundEState", [(round_state, index) for index, round_state in enumerate(ROUND_STATES)])
FIRST_BIDDING_STATE =        RoundEState["FIRST_BIDDING"]
DEALER_DISCARD_STATE =       RoundEState["DEALER_DISCARD"]
DECIDING_GOING_ALONE_STATE = RoundEState["DECIDING_GOING_ALONE"]
SECOND_BIDDING_STATE =       RoundEState["SECOND_BIDDING"]
CHOOSING_ESUIT_STATE =       RoundEState["CHOOSING_ESUIT"]
PLAYING_STATE =              RoundEState["PLAYING"]


ACTIONS = ["PLAY_CARD_1", "PLAY_CARD_2", "PLAY_CARD_3", "PLAY_CARD_4",
           "ORDER_UP", "PASS",
           "GO_ALONE", "DONT_GO_ALONE",
           "CHOOSE_SPADES", "CHOOSE_CLUBS", "CHOOSE_HEARTS", "CHOOSE_DIAMONDS"
           ]
EAction = IntEnum("EAction", [(action_name, index) for index, action_name in enumerate(ACTIONS)])
PLAY_CARD_1 =     EAction["PLAY_CARD_1"]
PLAY_CARD_2 =     EAction["PLAY_CARD_2"]
PLAY_CARD_3 =     EAction["PLAY_CARD_3"]
PLAY_CARD_4 =     EAction["PLAY_CARD_4"]
ORDER_UP =        EAction["ORDER_UP"]
PASS =            EAction["PASS"]
GO_ALONE =        EAction["GO_ALONE"]
DONT_GO_ALONE =   EAction["DONT_GO_ALONE"]
CHOOSE_SPADES =   EAction["CHOOSE_SPADES"]
CHOOSE_CLUBS =    EAction["CHOOSE_CLUBS"]
CHOOSE_HEARTS =   EAction["CHOOSE_HEARTS"]
CHOOSE_DIAMONDS = EAction["CHOOSE_DIAMONDS"]

PLAY_CARD_ACTIONS = [PLAY_CARD_1, PLAY_CARD_2, PLAY_CARD_3, PLAY_CARD_4]


suit_choice_action_to_esuit_map = {
    EAction["CHOOSE_SPADES"]: ESuit["SPADES"],
    EAction["CHOOSE_CLUBS"]: ESuit["CLUBS"],
    EAction["CHOOSE_HEARTS"]: ESuit["HEARTS"],
    EAction["CHOOSE_DIAMONDS"]: ESuit["DIAMONDS"]
}
def suit_choice_action_to_esuit(suit_choice_action):
    return suit_choice_action_to_esuit_map[suit_choice_action]


class Round:
    def __init__(self, last_dealer=None):
        self.finished = False
        self.trick_wins = [0] * 2
        self.round_points = [0] * 2
        self.trick_number = 0

        self.deck = Deck()
        self.dealer = get_clockwise_player(last_dealer) if last_dealer else EPlayer(random.randint(0, 3))

        self.current_player = get_clockwise_player(self.dealer)

        self.hands = [list([ECard(npint) for npint in array]) for array in np.array_split(self.deck.draw_ecards(16), 4)]

        self.upcard = ECard(self.deck.draw_ecards(1)[0])
        self.estate = FIRST_BIDDING_STATE

        self.trump_esuit = None
        self.maker = None
        self.going_alone = False

        self.played_ecards = []
        self.past_actions = []

    def get_actions(self) -> set[IntEnum]:
        if self.finished:
            return set()
        elif self.estate == FIRST_BIDDING_STATE:
            return {ORDER_UP, PASS}
        elif self.estate == DECIDING_GOING_ALONE_STATE:
            return {GO_ALONE, DONT_GO_ALONE}
        elif self.estate == SECOND_BIDDING_STATE:
            return {ORDER_UP, PASS}
        elif self.estate == CHOOSING_ESUIT_STATE:
            return {CHOOSE_SPADES, CHOOSE_CLUBS, CHOOSE_HEARTS, CHOOSE_DIAMONDS}
        elif self.estate == DEALER_DISCARD_STATE:
            return set(PLAY_CARD_ACTIONS)
        elif self.estate == PLAYING_STATE:
            current_hand = self.hands[self.current_player]

            # If no cards have been played, any remaining card can be played
            if len(self.played_ecards) == 0:
                actions = set()
                for i, ecard in enumerate(current_hand):
                    if ecard is not None:
                        actions.add(PLAY_CARD_ACTIONS[i])
                return actions

            # Determine which cards can be played according to the trump suit and led suit
            led_suit = get_ecard_esuit(self.played_ecards[0])

            is_valid = [False] * 4
            for i in range(4):
                ecard = current_hand[i]
                if ecard is None:
                    continue

                if get_ecard_esuit(ecard) == led_suit or get_ecard_esuit(ecard) == self.trump_esuit or (get_ecard_erank(ecard) == JACK and get_ecard_esuit(ecard) == get_same_color_esuit(self.trump_esuit)):
                    is_valid[i] = True

            # Allow the player to play valid cards (otherwise any remaining card if the player has no valid cards)
            if any(is_valid):
                actions = set()
                for i in range(4):
                    if is_valid[i]:
                        actions.add(PLAY_CARD_ACTIONS[i])
                return actions
            else:
                actions = set()
                for i, ecard in enumerate(current_hand):
                    if ecard is not None:
                        actions.add(PLAY_CARD_ACTIONS[i])
                return actions

    def take_action(self, action: IntEnum):
        assert action in self.get_actions(), f"The action {action} is not a legal move!"
        assert not self.finished, "No actions can be taken after the trick is finished!"

        self.past_actions.append((self.current_player, action, self.estate))

        if self.estate == FIRST_BIDDING_STATE:
            if action == ORDER_UP:
                self.trump_esuit = get_ecard_esuit(self.upcard)

                self.maker = self.current_player

                self.current_player = self.dealer
                self.estate = DEALER_DISCARD_STATE

            if action == PASS:
                if self.current_player == self.dealer:
                    self.estate = SECOND_BIDDING_STATE

                self.current_player = get_clockwise_player(self.current_player)

        elif self.estate == DEALER_DISCARD_STATE:
            replace_index = PLAY_CARD_ACTIONS.index(action)
            self.hands[self.current_player][replace_index] = self.upcard

            self.current_player = self.maker
            self.estate = DECIDING_GOING_ALONE_STATE

        elif self.estate == SECOND_BIDDING_STATE:
            if action == ORDER_UP:
                self.maker = self.current_player
                self.estate = CHOOSING_ESUIT_STATE

            if action == PASS:
                if self.current_player == self.dealer:
                    # If everyone passes for both rounds, reshuffle and do a new trick
                    self.finished = True

                self.current_player = get_clockwise_player(self.current_player)

        elif self.estate == CHOOSING_ESUIT_STATE:
            self.trump_esuit = suit_choice_action_to_esuit(action)

            self.estate = DECIDING_GOING_ALONE_STATE

        elif self.estate == DECIDING_GOING_ALONE_STATE:
            self.going_alone = action == GO_ALONE

            self.maker = self.current_player
            self.current_player = get_clockwise_player(self.dealer)
            self.estate = PLAYING_STATE

        elif self.estate == PLAYING_STATE:
            card_index = PLAY_CARD_ACTIONS.index(action)
            ecard = self.hands[self.current_player][card_index]
            self.hands[self.current_player][card_index] = None
            self.played_ecards.append(ecard)

            if len(self.played_ecards) == 4:
                self.end_trick()

                if self.trick_number == 4:
                    self.score_round()
                    self.finished = True
            else:
                self.current_player = get_clockwise_player(self.current_player)

    def get_current_player(self) -> IntEnum:
        return self.current_player

    def end_trick(self):
        # Determine the winner of the trick
        led_esuit = get_ecard_esuit(self.played_ecards[0])
        card_scores = [get_ecard_score(ecard, self.trump_esuit, led_esuit) for ecard in self.played_ecards]
        player_scores = [card_scores[(i + self.dealer + 3) % 4] for i in range(4)]  # TODO: make sure this is correct

        if self.going_alone:
            # Ignore the scores of teammates of players going alone
            player_scores[get_teammate(self.maker)] = 0

        highest_card_score = max(player_scores)
        winning_player = EPlayer(player_scores.index(highest_card_score))

        winning_team_index = eplayer_to_team_index(winning_player)

        self.trick_wins[winning_team_index] += 1

        # Set up the next trick
        self.trick_number += 1
        self.played_ecards = []
        self.current_player = winning_player

    def score_round(self):
        maker_team_index = eplayer_to_team_index(self.maker)
        defender_team_index = get_other_team_index(maker_team_index)

        if self.trick_wins[maker_team_index] == 5:
            self.round_points[maker_team_index] = 2

            # If the maker goes alone and wins all 5 tricks they get 4 points
            if self.going_alone:
                self.round_points[maker_team_index] = 4

        elif self.trick_wins[maker_team_index] >= 3:
            self.round_points[maker_team_index] = 1

        # The defenders "Eurchre" the makers if they win 3 or more
        if self.trick_wins[defender_team_index] >= 3:
            self.round_points[defender_team_index] = 2