from euchre.round import Round

from play_only_agent import PlayOnlyAgent
from fully_random_agent import FullyRandomAgent
from random_card_agent import RandomCardAgent

def compare_agents(agents, teammates=None, round_count=200):
    if teammates is None:
        teammates = agents

    wins = [0] * 2
    losses = [0] * 2
    ties = 0

    total_points = [0] * 2
    total_trick_wins = [0] * 2

    player_agents = [agents[0], agents[1], teammates[0], teammates[1]]

    for i in range(round_count):
        round = Round()
        while not round.finished:
            current_agent = player_agents[int(round.get_current_player())]
            current_agent.play(round)
        
        trick_wins = round.trick_wins
        round_points = round.round_points

        if sum(round_points) == 0:
            ties += 1
        else:
            if round_points[0] > round_points[1]:
                wins[0] += 1
                losses[1] += 1
            else:
                wins[1] += 1
                losses[0] += 1
            
            total_trick_wins[0] += trick_wins[0]
            total_trick_wins[1] += trick_wins[1]

            total_points[0] += round_points[0]
            total_points[1] += round_points[1]
    
    print(f"\nComparing {agents[0]} and {agents[1]} over {round_count} rounds:")

    def print_details(index):
        prefix = "First" if index == 0 else "Second"
        other_index = 1 if index == 0 else 0
        print(f"{prefix} Agent ({agents[index]}):")
        print(f"\t> Win Percentage (Ignoring Ties): {wins[index] / (wins[index] + losses[index]) * 100:.2f}")
        print(f"\t> Tie Percentage: {ties / (round_count) * 100:.2f}")
        print(f"\t> Average Points (IT): {total_points[index] / (wins[index] + losses[index]):.3f}")
        print(f"\t> Average Points Minus Opponent Points (IT): {(total_points[index] - total_points[other_index]) / (wins[index] + losses[index]):.3f}")
        print(f"\t> Average Trick Wins (IT): {total_trick_wins[index] / (wins[index] + losses[index]):.3f}")

    print_details(0)
    print()
    print_details(1)

if __name__ == "__main__":
    play_only_agent = PlayOnlyAgent()
    random_card_agent = RandomCardAgent()
    fully_random_agent = FullyRandomAgent()

    compare_agents([play_only_agent, random_card_agent], teammates=None, round_count=10000)
    print("------")
    compare_agents([play_only_agent, fully_random_agent], teammates=None, round_count=10000)