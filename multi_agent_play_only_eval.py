import torch
import numpy as np
from multi_agent_rl.multi_agent_play_only_env import EuchreMultiAgentEnv
from multi_agent_rl.multi_agent_play_only_train import PolicyNet

from bidding_hueristics import (
    should_order_up,
    should_call_suit_second_round,
    should_go_alone,
    get_suit_choice_action,
)

from euchre.round import (
    ORDER_UP, PASS, PLAYING_STATE,
    GO_ALONE, DONT_GO_ALONE,
    CHOOSING_ESUIT_STATE, SECOND_BIDDING_STATE,
    FIRST_BIDDING_STATE, DECIDING_GOING_ALONE_STATE
)

from euchre.players import PLAYERS, EPlayer, get_teammate
from euchre.deck import get_ecard_esuit

# ----------------------------------------------------------
#    Bidding heuristic wrapper (calls your heuristic logic)
# ----------------------------------------------------------

def take_bidding_action(env):
    round = env.round
    player = round.get_current_player()
    hand = round.hands[player]
    upcard = round.upcard
    is_dealer = (player == round.dealer)

    estate = round.estate
    
    # FIRST BIDDING ROUND (ORDER UP?)
    if estate == FIRST_BIDDING_STATE:
        if should_order_up(hand, upcard, is_dealer):
            return ORDER_UP
        return PASS

    # SECOND BIDDING ROUND (CHOOSE SUIT OR PASS)
    elif estate == SECOND_BIDDING_STATE:
        should_call, suit = should_call_suit_second_round(hand, upcard, is_dealer)
        if should_call:
            return get_suit_choice_action(suit)
        return PASS

    # PLAYER CHOOSES SUIT
    elif estate == CHOOSING_ESUIT_STATE:
        _, suit = should_call_suit_second_round(hand, upcard, is_dealer)
        return get_suit_choice_action(suit)

    # DECIDING GOING ALONE
    elif estate == DECIDING_GOING_ALONE_STATE:
        trump_suit = round.trump_esuit
        if should_go_alone(hand, trump_suit):
            return GO_ALONE
        return DONT_GO_ALONE

    raise RuntimeError("Unknown bidding state encountered.")

# ----------------------------------------------------------
#          Evaluate the trained RL policy 
# ----------------------------------------------------------

def evaluate_policy(model_path, num_episodes=500):
    env = EuchreMultiAgentEnv()

    # Get obs/action dimensions
    env.reset()
    first_agent = env.possible_agents[0]
    obs_dim = len(env.observe(first_agent)["observation"])
    act_dim = env.action_spaces[first_agent].n

    # Load the model
    policy = PolicyNet(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    team0_points = 0
    team1_points = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()

        done = False

        while not all(env.terminations.values()):
            agent = env.agent_selection
            player_idx = PLAYERS.index(agent)
            eplayer = EPlayer(player_idx)

            # If we are in bidding, use heuristics
            if env.round.estate != PLAYING_STATE:
                action = take_bidding_action(env)
                obs, rewards, terms, truncs, infos = env.step(action)
                continue

            # ------------------------------
            # RL policy for PLAYING_STATE
            # ------------------------------
            obs_vec = obs[agent]["observation"]
            action_mask = obs[agent]["action_mask"]

            obs_t = torch.tensor(obs_vec, dtype=torch.float32)

            # Policy may return (logits, value) or just logits
            output = policy(obs_t)
            logits = output[0] if isinstance(output, tuple) else output

            # Mask invalid actions
            logits_masked = logits.clone()
            invalid = (torch.tensor(action_mask) == 0)
            logits_masked[invalid] = -1e9

            action = torch.argmax(logits_masked).item()

            obs, rewards, terms, truncs, infos = env.step(action)

        # End of round: update points
        team0_points += env.round.round_points[0]
        team1_points += env.round.round_points[1]

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes} complete. "
                  f"Team0={team0_points}, Team1={team1_points}")

    print("=======================================")
    print("           FINAL RESULTS")
    print("=======================================")
    print(f"Team 0 Points: {team0_points}")
    print(f"Team 1 Points: {team1_points}")
    print(f"Team 0 win rate: {team0_points / (team0_points + team1_points + 1e-9):.3f}")

    return team0_points, team1_points


if __name__ == "__main__":
    evaluate_policy("euchre_models/policy_latest.pt", num_episodes=500)
