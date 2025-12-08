"""
Evaluation script: PlayOnlyAgent (RL) vs POMCP
"""

import argparse
from tqdm import tqdm
from dataclasses import dataclass

from euchre.round import Round
from euchre.players import EPlayer, eplayer_to_team_index

from play_only_agent import PlayOnlyAgent
from pomdp_agent import EuchrePOMDPAgent


@dataclass
class Metrics:
    # Round results
    team0_wins: int = 0
    team1_wins: int = 0
    team0_points: int = 0
    team1_points: int = 0
    ties: int = 0
    total_rounds: int = 0
    
    # Maker/defender
    team0_rounds_as_maker: int = 0
    team0_wins_as_maker: int = 0
    team0_points_as_maker: int = 0
    team1_rounds_as_maker: int = 0
    team1_wins_as_maker: int = 0
    team1_points_as_maker: int = 0
    
    team0_euchres: int = 0
    team1_euchres: int = 0
    
    # Tricks
    team0_tricks: int = 0
    team1_tricks: int = 0
    
    def print_report(self, team0_name="RL", team1_name="POMCP"):
        print("\n" + "=" * 60)
        print(f"EVALUATION: {team0_name} (Team 0) vs {team1_name} (Team 1)")
        print("=" * 60)
        
        total_decided = self.team0_wins + self.team1_wins
        
        print(f"\n{'ROUND RESULTS':-^60}")
        print(f"Total rounds: {self.total_rounds} ({self.ties} ties)")
        print(f"{team0_name} wins: {self.team0_wins} ({100*self.team0_wins/total_decided:.1f}%)")
        print(f"{team1_name} wins: {self.team1_wins} ({100*self.team1_wins/total_decided:.1f}%)")
        print(f"Total points - {team0_name}: {self.team0_points}, {team1_name}: {self.team1_points}")
        
        print(f"\n{'MAKER PERFORMANCE':-^60}")
        if self.team0_rounds_as_maker > 0:
            wr = 100 * self.team0_wins_as_maker / self.team0_rounds_as_maker
            ppg = self.team0_points_as_maker / self.team0_rounds_as_maker
            print(f"{team0_name} as maker: {self.team0_wins_as_maker}/{self.team0_rounds_as_maker} "
                  f"({wr:.1f}%) | {ppg:.2f} pts/round")
        else:
            print(f"{team0_name} as maker: 0 rounds")
            
        if self.team1_rounds_as_maker > 0:
            wr = 100 * self.team1_wins_as_maker / self.team1_rounds_as_maker
            ppg = self.team1_points_as_maker / self.team1_rounds_as_maker
            print(f"{team1_name} as maker: {self.team1_wins_as_maker}/{self.team1_rounds_as_maker} "
                  f"({wr:.1f}%) | {ppg:.2f} pts/round")
        else:
            print(f"{team1_name} as maker: 0 rounds")
        
        print(f"\n{'DEFENDER PERFORMANCE (EUCHRES)':-^60}")
        if self.team1_rounds_as_maker > 0:
            wr = 100 * self.team0_euchres / self.team1_rounds_as_maker
            print(f"{team0_name} euchres: {self.team0_euchres}/{self.team1_rounds_as_maker} ({wr:.1f}%)")
        if self.team0_rounds_as_maker > 0:
            wr = 100 * self.team1_euchres / self.team0_rounds_as_maker
            print(f"{team1_name} euchres: {self.team1_euchres}/{self.team0_rounds_as_maker} ({wr:.1f}%)")
        
        print(f"\n{'TRICKS':-^60}")
        if total_decided > 0:
            print(f"{team0_name} tricks: {self.team0_tricks} ({self.team0_tricks/total_decided:.2f}/round)")
            print(f"{team1_name} tricks: {self.team1_tricks} ({self.team1_tricks/total_decided:.2f}/round)")
        
        print("\n" + "=" * 60)


def run_evaluation(
    num_rounds: int = 1000,
    num_particles: int = 500,
    num_simulations: int = 200,
):
    print("Loading RL agent...")
    rl_agent = PlayOnlyAgent()
    
    print("Creating POMCP agents...")

    import numpy as np
    data = np.load('play_only_euchre_agent_bidding_weights_200000.npz')
    order_weights = data["order_weights"]
    order_bias = data["order_bias"]
    order_alone_weights = data["order_alone_weights"]
    order_alone_bias = data["order_alone_bias"]
    # Create agents for Team 0 (Player 0 and Player 2)
    agent_player0 = EuchrePOMDPAgent(
        player=EPlayer.Player1_Team1,
        order_weights=order_weights,
        order_bias=order_bias,
        order_alone_weights=order_alone_weights,
        order_alone_bias=order_alone_bias,
        num_particles=num_particles,
        num_simulations=num_simulations # higher is better but slower (200 = 7.5s per game on my machine)
    )

    agent_player2 = EuchrePOMDPAgent(
        player=EPlayer.Player3_Team1,
        order_weights=order_weights,
        order_bias=order_bias,
        order_alone_weights=order_alone_weights,
        order_alone_bias=order_alone_bias,
        num_particles=num_particles,
        num_simulations=num_simulations
    )

    pomcp_agents = {
        EPlayer.Player1_Team1: agent_player0,
        EPlayer.Player3_Team1: agent_player2
    }
    
    # Minimal game-like object for POMCP
    class FakeGame:
        def __init__(self, round_obj):
            self.round = round_obj
            self.finished = False
            self.team_points = [0, 0]
        def get_current_player(self):
            return self.round.get_current_player()
        def get_actions(self):
            return self.round.get_actions()
    
    metrics = Metrics()
    
    print(f"\nRunning {num_rounds} rounds...\n")
    
    for _ in tqdm(range(num_rounds)):
        round_obj = Round()
        fake_game = FakeGame(round_obj)
        
        # Reset POMCP beliefs
        for agent in pomcp_agents.values():
            agent.belief_initialized = False
        
        while not round_obj.finished:
            current_player = round_obj.get_current_player()
            team = eplayer_to_team_index(current_player)
            
            if team == 0:
                rl_agent.play(round_obj)
            else:
                action = pomcp_agents[current_player].get_action(fake_game)
                round_obj.take_action(action)
        
        # Record metrics
        record_round_metrics(metrics, round_obj)
    
    metrics.print_report()
    return metrics


def record_round_metrics(metrics: Metrics, round_obj):
    metrics.total_rounds += 1
    
    points = round_obj.round_points
    trick_wins = round_obj.trick_wins
    maker = round_obj.maker
    
    # Handle ties (everyone passed)
    if sum(points) == 0:
        metrics.ties += 1
        return
    
    # Track points and tricks
    metrics.team0_points += points[0]
    metrics.team1_points += points[1]
    metrics.team0_tricks += trick_wins[0]
    metrics.team1_tricks += trick_wins[1]
    
    # Determine round winner
    if points[0] > points[1]:
        metrics.team0_wins += 1
    else:
        metrics.team1_wins += 1
    
    # Maker/defender stats
    if maker is None:
        return
    
    maker_team = eplayer_to_team_index(maker)
    
    if maker_team == 0:
        metrics.team0_rounds_as_maker += 1
        if points[0] > 0:
            metrics.team0_wins_as_maker += 1
            metrics.team0_points_as_maker += points[0]
        else:
            metrics.team1_euchres += 1
    else:
        metrics.team1_rounds_as_maker += 1
        if points[1] > 0:
            metrics.team1_wins_as_maker += 1
            metrics.team1_points_as_maker += points[1]
        else:
            metrics.team0_euchres += 1


if __name__ == "__main__":
    rounds = 100
    # particles = [10,100,1000,10000]
    # simulations = [10,50,100,200, 500, 1000]
    # for p in particles:
    #     print(f"\n=== Evaluating with {p} particles ===")
    #     run_evaluation(
    #         num_rounds=rounds,
    #         num_particles=p,
    #         num_simulations=100,
    #     )
    # for s in simulations:
    #     print(f"\n=== Evaluating with {s} simulations ===")
    #     run_evaluation(
    #         num_rounds=rounds,
    #         num_particles=500,
    #         num_simulations=s,
    #     )
    run_evaluation(
        num_rounds=100,
        num_particles=500,
        num_simulations=1,
    )