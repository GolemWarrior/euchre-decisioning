from pomdp.particle_filter import ParticleFilter
from euchre.deck import ECard, ESuit
import numpy as np

NUM_PLAYERS = 4


# ============================================================
# Belief-printing utilities
# ============================================================

def print_beliefs_for_player(
    pf: ParticleFilter,
    player_index: int,
    top_n: int = 15,
    show_prob: bool = True,
) -> None:
    """
    Print the belief over a single player's hand (using PARTICLE WEIGHTS).
    """
    counts = {card: 0.0 for card in ECard}

    for w, ws in zip(pf.weights, pf.particles):
        for c in ws.hands[player_index]:
            counts[c] += w

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    total_mass = sum(counts.values())

    print(f"--- Belief over Player {player_index}'s hand ---")
    for c, mass in sorted_counts[:top_n]:
        if mass <= 0.0:
            continue
        if show_prob:
            print(f"{c.name:<20} {mass:.3f}")
        else:
            pseudo_count = mass * pf.num_particles
            print(f"{c.name:<20} {pseudo_count:.0f}")
    print(f"(Total belief mass = {total_mass:.3f})\n")


def print_all_beliefs(pf: ParticleFilter, label: str) -> None:
    print("\n========================")
    print(f" BELIEFS: {label}")
    print("========================\n")

    for player in range(NUM_PLAYERS):
        print_beliefs_for_player(pf, player_index=player, top_n=15, show_prob=True)


# ============================================================
# Simulation loop with rejuvenation
# ============================================================

def simulate_trick(
    pf: ParticleFilter,
    plays,
    my_hand,
    rejuvenate_rate: float = 0.05,
    rejuvenate_turns: int = 2,
) -> None:
    """
    Simulate a trick with multiple moves.

    plays: list of tuples (player_index, card_played, fail_follow_suit_or_None)
    rejuvenate_rate: probability of swapping a random unseen card
    rejuvenate_turns: rejuvenate for this many initial turns (early phase)
    """
    pf.initialize_uniform(my_hand=my_hand)
    print_all_beliefs(pf, "Initial uniform distribution")

    for turn, (player, card, fail_suit) in enumerate(plays, 1):
        print("\n========================")
        print(f" TURN {turn}: Player {player}")
        print("========================\n")

        # -------------------------
        # Fail-to-follow observation
        # -------------------------
        if fail_suit is not None:
            pf.observe_fail_follow(player=player, suit=fail_suit)
            print_all_beliefs(
                pf,
                f"After observing Player {player} FAILS to follow {fail_suit.name}",
            )

        # -------------------------
        # Play observation
        # -------------------------
        pf.observe_play(player=player, card=card)
        pf.resample()
        print_all_beliefs(
            pf,
            f"After Player {player} PLAYS {card.name}",
        )

        # -------------------------
        # OPTIONAL REJUVENATION
        # -------------------------
        if turn <= rejuvenate_turns:
            print("\n--- Running rejuvenation step ---")
            before_ess = (1.0 / np.sum(pf.weights**2))
            pf.rejuvenate(rate=rejuvenate_rate)
            after_ess = (1.0 / np.sum(pf.weights**2))
            print(f"Rejuvenation complete. ESS before={before_ess:.1f}, after={after_ess:.1f}")

    # ============================================================
    # Summary after all turns
    # ============================================================
    print("\nAll turns complete.")

    valid_indices = [i for i, w in enumerate(pf.weights) if w > 0.0]
    print("Valid particles (weight > 0):", len(valid_indices))
    print("\n=== Hand size summary across all particles ===")
    for p in range(4):
        sizes = [len(ws.hands[p]) for ws in pf.particles]
        unique_sizes = sorted(set(sizes))
        print(f"Player {p}: unique hand sizes = {unique_sizes}")

    if valid_indices:
        rand_idx = np.random.choice(valid_indices)
        rand_ws = pf.particles[rand_idx]
        print("\nRandom valid particle:")
        for p in range(NUM_PLAYERS):
            print(f"Player {p} hand:", [c.name for c in rand_ws.hands[p]])
    else:
        print("No valid particles found!")


# ============================================================
# Main entry point
# ============================================================

def main():
    pf = ParticleFilter(num_particles=5000)

    # Our hand (Player 0)
    my_hand = {
        ECard.SPADES_9,
        ECard.CLUBS_JACK,
        ECard.HEARTS_KING,
        ECard.DIAMONDS_10,
        ECard.SPADES_KING,
    }

    # Define a trick: (player_index, card_played, fail_follow_suit_or_None)
    plays = [
        (1, ECard.CLUBS_10, ESuit.HEARTS),
        (2, ECard.DIAMONDS_KING, None),
        (3, ECard.SPADES_10, None),
        (0, ECard.SPADES_9, None),
    ]

    simulate_trick(
        pf,
        plays,
        my_hand=my_hand,
        rejuvenate_rate=0.05,
        rejuvenate_turns=2,
    )


if __name__ == "__main__":
    main()
