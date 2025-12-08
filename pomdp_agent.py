from tqdm import tqdm
import random
from euchre.game import Game
from euchre.game import Game
from euchre.round import (
    EAction,
    FIRST_BIDDING_STATE,
    DEALER_DISCARD_STATE,
    DECIDING_GOING_ALONE_STATE,
    SECOND_BIDDING_STATE,
    CHOOSING_ESUIT_STATE,
    PLAYING_STATE,
    PLAY_CARD_ACTIONS
)
from euchre.players import EPlayer, eplayer_to_team_index
from euchre.deck import ECard, get_ecard_esuit
from mcts import POMCPPlanner

from learn_bidding import get_best_bidding_action, BIDDING_ESTATES


class EuchrePOMDPAgent:
    """
    Agent that plays Euchre using POMCP for card play and heuristics for bidding.

    This agent:
    - Uses bidding heuristics during the bidding phases
    - Uses POMCP planning during card play
    - Maintains a particle filter for belief state tracking
    - Controls a single player (can be extended to control a team)
    """

    def __init__(
        self,
        player: EPlayer,
        order_weights,
        order_bias,
        order_alone_weights,
        order_alone_bias,
        num_particles: int = 500,
        num_simulations: int = 1000,
        exploration_constant: float = 10.0
    ):
        """
        Initialize the POMDP agent.

        Args:
            player: The player this agent controls (EPlayer enum)
            num_particles: Number of particles for belief state representation
            num_simulations: Number of MCTS simulations per action selection
            exploration_constant: UCB exploration parameter
            max_depth: Maximum simulation depth
        """
        self.player = player
        self.team_index = eplayer_to_team_index(player)

        self.order_weights = order_weights
        self.order_bias = order_bias
        self.order_alone_weights = order_alone_weights
        self.order_alone_bias = order_alone_bias

        # POMCP planner for card play
        self.planner = POMCPPlanner(
            player=player,
            num_particles=num_particles,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant
        )

        # Track whether belief has been initialized
        self.belief_initialized = False

    def get_action(self, game: Game):
        """
        Main interface: get an action for the current game state.

        Args:
            game: Current game state

        Returns:
            Action to take (EAction)
        """
        round_state = game.round.estate

        # Reset belief flag when round finishes
        if game.round.finished:
            self.belief_initialized = False

        # Route to appropriate decision method based on game state
        # Use RL bidding for all bidding states
        if round_state in BIDDING_ESTATES:
            return get_best_bidding_action(
                game.round,
                self.order_weights,
                self.order_bias,
                self.order_alone_weights,
                self.order_alone_bias
            )

        elif round_state == PLAYING_STATE:
            if not self.belief_initialized:
                self._initialize_belief(game)
                self.belief_initialized = True
            return self._get_playing_action(game)

        else:
            raise ValueError(f"Unknown round state: {round_state}")

    def _initialize_belief(self, game: Game):
        """Initialize the particle filter with current hand."""
        my_hand = set(card for card in game.round.hands[self.player] if card is not None)
        self.planner.initialize(my_hand)

    def _get_playing_action(self, game: Game):
        """
        Choose which card to play using POMCP planner.
        """
        return self.planner.select_action(game)
    
def test_agent(print_output: bool = True):
    game = Game()
    import numpy as np
    data = np.load('play_only_euchre_agent_bidding_weights_200000.npz')
    order_weights = data["order_weights"]
    order_bias = data["order_bias"]
    order_alone_weights = data["order_alone_weights"]
    order_alone_bias = data["order_alone_bias"]
    # Create agents for Team 0 (Player 0 and Player 2)
    agent_player0 = EuchrePOMDPAgent(
        player=EPlayer.Player0_Team0,
        order_weights=order_weights,
        order_bias=order_bias,
        order_alone_weights=order_alone_weights,
        order_alone_bias=order_alone_bias,
        num_particles=500,
        num_simulations=200 # higher is better but slower (200 = 7.5s per game on my machine)
    )

    agent_player2 = EuchrePOMDPAgent(
        player=EPlayer.Player2_Team0,
        order_weights=order_weights,
        order_bias=order_bias,
        order_alone_weights=order_alone_weights,
        order_alone_bias=order_alone_bias,
        num_particles=500,
        num_simulations=200
    )

    agents = {
        EPlayer.Player0_Team0: agent_player0,
        EPlayer.Player2_Team0: agent_player2
    }
    if print_output:
        print("=" * 60)
        print("EUCHRE GAME: POMCP Agents (Team 0) vs Random (Team 1)")
        print("=" * 60)
        print()

    game_turn = 0
    round_number = 1

    while not game.finished:
        current_player = game.get_current_player()
        legal_actions = game.get_actions()

        # Print game state
        if print_output:
            if game.round.estate == FIRST_BIDDING_STATE and game.round.current_player == game.round.dealer + 1 or (game.round.current_player == 0 and game.round.dealer == 3):
                print(f"\n{'='*60}")
                print(f"ROUND {round_number}")
                print(f"{'='*60}")
                print(f"Dealer: {game.round.dealer.name}")
                print(f"Upcard: {game.round.upcard.name}")
                print(f"Team 0 Score: {game.team_points[0]} | Team 1 Score: {game.team_points[1]}")
                round_number += 1

        # Show current turn info
        team = "Team 0 (POMCP)" if current_player in agents else "Team 1 (Random)"
        state_name = game.round.estate.name
        if print_output:
            print(f"Turn {game_turn}: {current_player.name} ({team}) - State: {state_name}")

        # Select action
        if current_player in agents:
            # POMCP agent
            action = agents[current_player].get_action(game)
            if print_output:
                print(f" -> POMCP selected: {action.name}")
        else:
            # Random agent
            action = random.choice(list(legal_actions))
            if print_output:
                print(f" -> Random selected: {action.name}")

        # Show card played if in playing state
        if game.round.estate == PLAYING_STATE and action in PLAY_CARD_ACTIONS:
            card_index = PLAY_CARD_ACTIONS.index(action)
            card = game.round.hands[current_player][card_index]
            if card and print_output:
                print(f"    Played: {card.name}")

        # Take action
        game.take_action(action)
        game_turn += 1

        # Show trick results
        if game.round.estate == PLAYING_STATE and len(game.round.played_ecards) == 0 and len(game.round.past_played_ecard_lists) > 0:
            trick_num = len(game.round.past_played_ecard_lists)
            if print_output:
                print(f"\nTrick {trick_num} complete!")
                print(f"Trick wins: Team 0: {game.round.trick_wins[0]} | Team 1: {game.round.trick_wins[1]}")

        # Show round results
        if game.round.finished and not game.finished and print_output:
            print(f"\n{'─'*60}")
            print(f"ROUND COMPLETE")
            print(f"Trump suit: {game.round.trump_esuit.name if game.round.trump_esuit else 'None'}")
            print(f"Tricks won: Team 0: {game.round.trick_wins[0]} | Team 1: {game.round.trick_wins[1]}")
            print(f"Points awarded: Team 0: +{game.round.round_points[0]} | Team 1: +{game.round.round_points[1]}")
            print(f"Total Score: Team 0: {game.team_points[0]} | Team 1: {game.team_points[1]}")
            print(f"{'─'*60}\n")

    # Game over
    if print_output:
        print(f"\n{'='*60}")
        print(f"GAME OVER")
        print(f"Winner: Team {game.winner}")
        print(f"Final Score: Team 0: {game.team_points[0]} | Team 1: {game.team_points[1]}")
        print(f"Total turns: {game_turn}")
        print(f"{'='*60}\n")

    return game.winner,game.team_points

def run_sims(n = 100):
    team0_wins = 0
    team1_wins = 0
    team0_points = 0
    team1_points = 0
    for _ in tqdm(range(n)):
        winner,points = test_agent(print_output=False)
        if winner == 0:
            team0_wins += 1
        else:
            team1_wins += 1
        team0_points += points[0]
        team1_points += points[1]
    print(f"After {n} games:")
    print(f"Team 0 (POMCP) wins: {team0_wins} ({team0_wins/n*100:.2f}%)")
    print(f"Team 1 (Random) wins: {team1_wins} ({team1_wins/n*100:.2f}%)")
    print(f"Average points per game - Team 0: {team0_points/n:.2f}, Team 1: {team1_points/n:.2f}")

if __name__ == "__main__":
    run_sims(10)
    #test_agent(print_output=True)