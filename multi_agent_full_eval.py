import random
import numpy as np
import torch

from euchre.round import (
    Round,
    PLAYING_STATE,
    PLAY_CARD_ACTIONS,
    ACTIONS,
    EAction,
)
from euchre.players import (
    EPlayer,
    eplayer_to_team_index,
    get_teammate,
)
from euchre.deck import DECK_SIZE, ECard

from state_encoding.multi_agent_play_only_rl import encode_state, encode_playable
from multi_agent_bidding_train import BiddingPolicy, encode_bidding_obs, encode_bidding_mask
from multi_agent_play_train import PlayPolicy, random_legal_bidding_action, random_legal_play_action


# -------------
#  Helpers
# -------------
def load_bidding_policy():
    dummy = Round()
    obs_example = encode_bidding_obs(dummy, EPlayer(0))
    obs_dim = len(obs_example)
    act_dim = len(ACTIONS)

    policy = BiddingPolicy(obs_dim, act_dim)
    state_dict = torch.load("euchre_models/bidding_policy_latest.pt", map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_play_policy():
    dummy = Round()
    # Move dummy into PLAYING_STATE similarly to play_train
    while dummy.estate != PLAYING_STATE and not dummy.finished:
        dummy.take_action(random_legal_bidding_action(dummy))
    if not dummy.finished and dummy.estate == PLAYING_STATE and dummy.get_actions():
        dummy.take_action(random.choice(list(dummy.get_actions())))

    obs_example = encode_state(dummy, agent_player=EPlayer(0))
    obs_dim = len(obs_example)
    act_dim = DECK_SIZE

    policy = PlayPolicy(obs_dim, act_dim)
    state_dict = torch.load("euchre_models/play_policy_latest.pt", map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def team0_bidding_action(round_obj: Round, bidding_policy: BiddingPolicy, player: EPlayer) -> EAction:
    obs_vec = encode_bidding_obs(round_obj, player)
    mask_vec = encode_bidding_mask(round_obj)

    obs_t = torch.tensor(obs_vec, dtype=torch.float32)
    mask_t = torch.tensor(mask_vec, dtype=torch.float32)

    logits = bidding_policy(obs_t)
    masked_logits = logits.clone()
    masked_logits[mask_t == 0] = -1e9

    action_idx = torch.argmax(masked_logits).item()
    action_enum = EAction(int(action_idx))

    if action_enum not in round_obj.get_actions():
        action_enum = random_legal_bidding_action(round_obj)

    return action_enum


def team0_play_action(round_obj: Round, play_policy: PlayPolicy, player: EPlayer) -> EAction:
    obs_vec = encode_state(round_obj, agent_player=player)
    mask_vec = encode_playable(round_obj, agent_player=player)

    obs_t = torch.tensor(obs_vec, dtype=torch.float32)
    logits = play_policy(obs_t)

    mask_t = torch.tensor(mask_vec, dtype=torch.float32)
    masked_logits = logits.clone()
    masked_logits[mask_t == 0] = -1e9

    card_idx = torch.argmax(masked_logits).item()
    chosen_card = ECard(card_idx)

    hand = round_obj.hands[int(player)]
    if chosen_card not in hand:
        return random_legal_play_action(round_obj)

    hand_pos = hand.index(chosen_card)
    action_enum = PLAY_CARD_ACTIONS[hand_pos]

    if action_enum not in round_obj.get_actions():
        return random_legal_play_action(round_obj)

    return action_enum


# -------------
#  Evaluation
# -------------
def evaluate(num_episodes=5000):
    bidding_policy = load_bidding_policy()
    play_policy = load_play_policy()

    team0_points = 0
    team1_points = 0

    for ep in range(num_episodes):
        round_obj = Round()

        while not round_obj.finished:
            # BIDDING
            if round_obj.estate != PLAYING_STATE:
                player = round_obj.get_current_player()
                team = eplayer_to_team_index(player)

                if team == 0:
                    action_enum = team0_bidding_action(round_obj, bidding_policy, player)
                else:
                    action_enum = random_legal_bidding_action(round_obj)

                round_obj.take_action(action_enum)
                continue

            # PLAYING
            player = round_obj.get_current_player()
            team = eplayer_to_team_index(player)

            # Going alone partner skipping
            if round_obj.going_alone and round_obj.maker is not None:
                if player == get_teammate(round_obj.maker):
                    action_enum = random_legal_play_action(round_obj)
                    if action_enum is None:
                        break
                    round_obj.take_action(action_enum)
                    continue

            if team == 0:
                action_enum = team0_play_action(round_obj, play_policy, player)
            else:
                action_enum = random_legal_play_action(round_obj)

            if action_enum is None:
                break

            round_obj.take_action(action_enum)

        team0_points += round_obj.round_points[0]
        team1_points += round_obj.round_points[1]

        if (ep + 1) % 100 == 0:
            wr = team0_points / (team0_points + team1_points + 1e-9)
            print(
                f"[EVAL {ep+1}/{num_episodes}] "
                f"Team0={team0_points}, Team1={team1_points}, WinRate={wr:.3f}"
            )

    print("=================================")
    print(" FINAL RESULTS ")
    print("=================================")
    print(f"Team 0 Points: {team0_points}")
    print(f"Team 1 Points: {team1_points}")
    print(f"Team 0 Win Rate: {team0_points / (team0_points + team1_points + 1e-9):.3f}")


if __name__ == "__main__":
    evaluate(num_episodes=2000)
