import random
import math
from typing import Dict, Set, Optional
from copy import deepcopy

from euchre.round import EAction, PLAYING_STATE
from euchre.deck import ECard, get_ecard_esuit
from euchre.players import EPlayer, eplayer_to_team_index
from pomdp.particle_filter import ParticleFilter


class MCTSNode:
    """Node in the search tree."""
    
    __slots__ = ['visit_count', 'action_values', 'action_visits', 'children']
    
    def __init__(self):
        self.visit_count: int = 0
        self.action_values: Dict[EAction, float] = {}
        self.action_visits: Dict[EAction, int] = {}
        self.children: Dict[EAction, 'MCTSNode'] = {}
    
    def ucb_score(self, action: EAction, c: float):
        """UCB1: Q(a) + c * sqrt(ln(N) / N(a))"""
        if action not in self.action_visits or self.action_visits[action] == 0:
            return float('inf')
        
        q = self.action_values[action]
        exploration = c * math.sqrt(math.log(self.visit_count) / self.action_visits[action])
        return q + exploration
    
    def select_action_ucb(self, legal_actions: Set[EAction], c: float):
        """Select action with highest UCB score."""
        return max(legal_actions, key=lambda a: self.ucb_score(a, c))
    
    def select_action_greedy(self, legal_actions: Set[EAction]):
        """Select action with highest Q value (for final selection)."""
        visited = [a for a in legal_actions if self.action_visits.get(a, 0) > 0]
        if not visited:
            return random.choice(list(legal_actions))
        return max(visited, key=lambda a: self.action_values[a])
    
    def update(self, action: EAction, reward: float):
        """Incremental mean update for action value."""
        if action not in self.action_values:
            self.action_values[action] = 0.0
            self.action_visits[action] = 0
        
        self.action_visits[action] += 1
        self.visit_count += 1
        
        # Incremental mean: Q = Q + (r - Q) / n
        n = self.action_visits[action]
        self.action_values[action] += (reward - self.action_values[action]) / n


class POMCPPlanner:
    """
    POMCP planner for a single Euchre player.
    
    Uses particle filter for belief state and MCTS for action selection.
    (bidding handled separately)
    (batteries not included)
    """
    
    def __init__(
        self,
        player: EPlayer,
        num_particles: int = 500,
        num_simulations: int = 100,
        exploration_constant: float = 1.5
    ):
        self.player = player
        self.team_index = eplayer_to_team_index(player)
        self.num_particles = num_particles
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        
        self.particle_filter = ParticleFilter(num_particles=num_particles)
        self.root: Optional[MCTSNode] = None
        
        # Track which actions we've already processed
        self._processed_action_count = 0
    
    def initialize(self, my_hand: Set[ECard]):
        """Initialize belief state at start of round."""
        self.particle_filter.initialize_uniform(my_hand)
        self.root = MCTSNode()
        self._processed_action_count = 0
    
    def select_action(self, game):
        """
        Select best action using POMCP search.
        
        Args:
            game: Current game state (Game object with round info)
        
        Returns:
            Best action according to MCTS
        """
        round_state = game.round
        
        # Update belief based on any new observations
        self._update_belief_from_game(round_state)
        
        # Reset search tree for each action
        self.root = MCTSNode()
        
        legal_actions = round_state.get_actions()
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Sample a particle
            particle = self._sample_particle()
            
            # Create determinized game state
            sim_round = self._create_sim_round(round_state, particle)
            
            # Run simulation
            self._simulate(self.root, sim_round, depth=0)
        
        # Select best action (greedy)
        return self.root.select_action_greedy(legal_actions)
    
    def _sample_particle(self):
        """Sample a world state from the particle filter."""
        idx = random.choices(
            range(self.num_particles),
            weights=self.particle_filter.weights
        )[0]
        return self.particle_filter.particles[idx]
    
    def _simulate(self, node: MCTSNode, sim_round, depth: int):
        """
        Run one MCTS simulation.
        
        Returns reward from perspective of self.player's team.
        """
        # Terminal check
        if sim_round.finished:
            return self._evaluate(sim_round)
        
        current_player = sim_round.get_current_player()
        legal_actions = sim_round.get_actions()
        
        # If not our turn, use rollout policy and continue
        if current_player != self.player:
            action = self._rollout_policy(legal_actions)
            sim_round.take_action(action)
            return self._simulate(node, sim_round, depth + 1)
        
        # Our turn: MCTS selection
        if node.visit_count == 0:
            # First visit: rollout
            action = self._rollout_policy(legal_actions)
            sim_round.take_action(action)
            reward = self._rollout(sim_round, depth + 1)
            node.update(action, reward)
            return reward
        
        # Selection via UCB
        action = node.select_action_ucb(legal_actions, self.exploration_constant)
        
        # Expansion: create child if needed
        if action not in node.children:
            node.children[action] = MCTSNode()
        
        child = node.children[action]
        sim_round.take_action(action)
        
        # Recurse
        reward = self._simulate(child, sim_round, depth + 1)
        
        # Backprop
        node.update(action, reward)
        
        return reward
    
    def _rollout(self, sim_round, depth: int):
        """Random rollout to terminal state."""
        while not sim_round.finished:
            legal_actions = sim_round.get_actions()
            action = self._rollout_policy( legal_actions)
            sim_round.take_action(action)
            depth += 1
        
        return self._evaluate(sim_round)
    
    def _rollout_policy(self, legal_actions: Set[EAction]):
        """
        Simple rollout policy. Random legal action.
        """
        return random.choice(list(legal_actions))
    
    def _evaluate(self, sim_round) -> float:
        """
        Evaluate terminal/leaf state.
        
        Returns value in [-1, 1] for our team.
        """
        if not sim_round.finished:
            # Mid-game evaluation based on tricks won
            our_tricks = sim_round.trick_wins[self.team_index]
            opp_tricks = sim_round.trick_wins[1 - self.team_index]
            return (our_tricks - opp_tricks) / 5.0
        
        # Round finished - evaluate based on points
        our_points = sim_round.round_points[self.team_index]
        opp_points = sim_round.round_points[1 - self.team_index]
        
        if our_points > opp_points:
            return 1.0
        elif opp_points > our_points:
            return -1.0
        return 0.0
    
    def _create_sim_round(self, real_round, particle):
        """
        Create a simulation round by combining real game state with sampled particle.
        
        The particle tells us what cards opponents hold.
        """
        sim = deepcopy(real_round)
        
        # Replace opponent hands with particle's hands
        for p in range(4):
            if p == self.player:
                continue  # Keep our actual hand
            
            # Get cards from particle for this player
            particle_cards = list(particle.hands[p])
            
            # Fill in non-None slots in sim.hands[p]
            card_idx = 0
            for i in range(len(sim.hands[p])):
                if sim.hands[p][i] is not None:
                    if card_idx < len(particle_cards):
                        sim.hands[p][i] = particle_cards[card_idx]
                        card_idx += 1
        
        return sim
    
    def _update_belief_from_game(self, round_state):
        """
        Update particle filter based on observed plays.
        
        Called before each action selection to sync belief with game state.
        Only processes actions we haven't seen yet.
        """
        # Only process new actions
        new_actions = round_state.past_actions[self._processed_action_count:]
        
        for player, action, state, card in new_actions:
            if state != PLAYING_STATE or card is None:
                self._processed_action_count += 1
                continue
            
            if player == self.player:
                # Still need to track that we processed it, but don't update belief
                self._processed_action_count += 1
                continue
            
            # Check for fail-to-follow by looking at current trick context
            # We need to find the led card for this trick
            led_suit = self._get_led_suit_for_trick(round_state, self._processed_action_count)
            if led_suit is not None:
                card_suit = get_ecard_esuit(card)
                if card_suit != led_suit:
                    self.particle_filter.observe_fail_follow(player, led_suit)
            
            # Observe the card play
            self.particle_filter.observe_play(player, card)
            self._processed_action_count += 1
        
        # Resample if needed
        ess = self.particle_filter.effective_sample_size()
        if ess < self.num_particles / 2:
            self.particle_filter.resample()
            self.particle_filter.rejuvenate(rate=0.1)
    
    def _get_led_suit_for_trick(self, round_state, action_index):
        """
        Determine what suit was led for the trick containing the action at action_index.
        Looks backwards through past_actions to find the first card of this trick.
        """
        # Count how many cards were played before this action
        cards_before = 0
        for i in range(action_index):
            _, _, state, card = round_state.past_actions[i]
            if state == PLAYING_STATE and card is not None:
                cards_before += 1
        
        # Which trick is this? (0-indexed)
        trick_num = cards_before // 4
        
        # Find the first card of this trick
        cards_seen = 0
        for _, _, state, card in round_state.past_actions:
            if state == PLAYING_STATE and card is not None:
                if cards_seen // 4 == trick_num:
                    # This is the lead card of the current trick
                    return get_ecard_esuit(card)
                cards_seen += 1
        
        return None
    
    def observe_action(self, player: EPlayer, card: Optional[ECard] = None):
        """
        Call this after each action in the game to keep belief state current.
        """
        if card is not None and player != self.player:
            self.particle_filter.observe_play(player, card)
