from pomdp.particle_filter import ParticleFilter
from euchre.deck import ECard, ESuit
import numpy as np

def print_beliefs(pf, label, player_index=1, top_n=20, show_prob=False):
    print(f"\n=== {label} ===")
    counts = {card: 0 for card in ECard}
    for ws in pf.particles:
        for c in ws.hands[player_index]:
            counts[c] += 1
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    total_particles = len(pf.particles)
    for c, count in sorted_counts[:top_n]:
        if show_prob:
            print(f"{c.name:<20} {count/total_particles:.2f}")
        else:
            print(f"{c.name:<20} {count}")

def simulate_trick(pf, plays, my_hand, player_to_watch=1, rejuvenate_rate=0.05):
    """
    Simulate a trick with multiple moves.
    
    plays: list of tuples (player_index, card_played, fail_follow_suit_or_None)
    """
    pf.initialize_uniform(my_hand=my_hand)
    print_beliefs(pf, "Initial uniform distribution", player_index=player_to_watch)

    for turn, (player, card, fail_suit) in enumerate(plays, 1):
        print(f"\n--- Turn {turn}: Player {player} ---")

        # Observe fail-to-follow if applicable
        if fail_suit is not None:
            pf.observe_fail_follow(player=player, suit=fail_suit)
            print_beliefs(pf, f"After observing Player {player} fails to follow {fail_suit.name}", player_index=player_to_watch)

        # Observe actual play
        pf.observe_play(player=player, card=card)
        pf.resample()
        print_beliefs(pf, f"After Player {player} plays {card.name}", player_index=player_to_watch)

        # Rejuvenate to maintain diversity
        # pf.rejuvenate(rate=rejuvenate_rate)
        # print_beliefs(pf, f"After rejuvenation", player_index=player_to_watch)

def main():
    pf = ParticleFilter(num_particles=2000)
    print("This is just the PF states for Player 1 (the one to the left of you)")
    # Our hand (Player 0)
    my_hand = {ECard.SPADES_9, ECard.CLUBS_JACK, ECard.HEARTS_KING,
               ECard.DIAMONDS_10, ECard.SPADES_KING}

    # Define a trick: (player_index, card_played, fail_follow_suit_or_None)
    plays = [
        (1, ECard.CLUBS_10, ESuit.HEARTS),  # Player 1: fails to follow hearts, then plays CLUBS_10
        (2, ECard.DIAMONDS_KING, None),     # Player 2: no fail-to-follow, plays DIAMONDS_KING
        (3, ECard.SPADES_10, None),         # Player 3: no fail-to-follow, plays SPADES_10
        (0, ECard.SPADES_9, None)           # Player 0 (you) plays SPADES_9
    ]

    simulate_trick(pf, plays, my_hand=my_hand, player_to_watch=1, rejuvenate_rate=0.05)
    print("\nAll turns complete.")

if __name__ == "__main__":
    main()
 