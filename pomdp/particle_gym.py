from pomdp.particle_filter import ParticleFilter
from euchre.deck import ECard, ESuit
import numpy as np

NUM_PLAYERS = 4

def print_beliefs_for_player(
    pf: ParticleFilter,
    player_index: int,
    top_n: int = 15,
    show_prob: bool = True,
) -> None:
    """
    Print the belief over a single player's hand using PARTICLE WEIGHTS.
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


def count_unique_hands(pf: ParticleFilter, player_index: int) -> int:
    """
    How many distinct hand configurations does this player have
    across all particles (as a diversity metric)?
    """
    hand_configs = set()
    for ws in pf.particles:
        # Sort by card enum value so sets are comparable
        hand_configs.add(tuple(sorted(c.value for c in ws.hands[player_index])))
    return len(hand_configs)


def simulate_trick(
    pf: ParticleFilter,
    plays,
    my_hand,
    rejuvenate_rate: float = 0.05,
) -> None:
    """
    Simulate a trick with multiple moves.

    plays: list of tuples (player_index, card_played, fail_follow_suit_or_None)
    """
    pf.initialize_uniform(my_hand=my_hand)
    print_all_beliefs(pf, "Initial uniform distribution")
    print(f"Initial ESS = {pf.effective_sample_size():.1f}")
    for p in range(NUM_PLAYERS):
        print(f"Initial unique hands for Player {p}: {count_unique_hands(pf, p)}")

    for turn, (player, card, fail_suit) in enumerate(plays, 1):
        print("\n========================")
        print(f" TURN {turn}: Player {player}")
        print("========================\n")

        # Observe fail-to-follow if applicable
        if fail_suit is not None:
            pf.observe_fail_follow(player=player, suit=fail_suit)
            print_all_beliefs(
                pf,
                f"After observing Player {player} FAILS to follow {fail_suit.name}",
            )
            print(f"ESS after fail-follow = {pf.effective_sample_size():.1f}")

        # Observe actual play
        pf.observe_play(player=player, card=card)
        print_all_beliefs(
            pf,
            f"After Player {player} PLAYS {card.name} (pre-resample)",
        )
        print(f"ESS before resample = {pf.effective_sample_size():.1f}")

        # Resample
        pf.resample()
        print_all_beliefs(
            pf,
            f"After Player {player} PLAYS {card.name} (post-resample)",
        )
        ess_after_resample = pf.effective_sample_size()
        print(f"ESS after resample = {ess_after_resample:.1f}")

        # Diversity before rejuvenation
        uniq_before = {
            p: count_unique_hands(pf, p) for p in range(NUM_PLAYERS)
        }

        # Rejuvenation=
        if rejuvenate_rate > 0.0:
            print("\n--- Running rejuvenation step ---")
            pf.rejuvenate(rate=rejuvenate_rate)
            ess_after_rejuv = pf.effective_sample_size()
            uniq_after = {
                p: count_unique_hands(pf, p) for p in range(NUM_PLAYERS)
            }
            print(
                f"Rejuvenation complete. ESS (before/after) = "
                f"{ess_after_resample:.1f} / {ess_after_rejuv:.1f}"
            )
            for p in range(NUM_PLAYERS):
                print(
                    f"Player {p}: unique hands before/after rejuvenation = "
                    f"{uniq_before[p]} / {uniq_after[p]}"
                )

    print("\nAll turns complete.")

    # "Valid" now means weight > 0
    valid_indices = [i for i, w in enumerate(pf.weights) if w > 0.0]
    print("Valid particles (weight > 0):", len(valid_indices))

    print("\n=== Hand size summary across all particles ===")
    for p in range(NUM_PLAYERS):
        sizes = [len(ws.hands[p]) for ws in pf.particles]
        unique_sizes = sorted(set(sizes))
        print(f"Player {p}: unique hand sizes = {unique_sizes}")

    if valid_indices:
        rand_idx = np.random.choice(valid_indices)
        rand_ws = pf.particles[rand_idx]
        print("\nRandom valid particle:")
        for p in range(NUM_PLAYERS):
            hand_names = [c.name for c in rand_ws.hands[p]]
            print(f"Player {p} hand: {sorted(hand_names)}")
    else:
        print("No valid particles found!")


def main():
    pf = ParticleFilter(num_particles=2000)

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
        (1, ECard.CLUBS_10, ESuit.HEARTS),  # Player 1: fails to follow hearts, then plays CLUBS_10
        (2, ECard.DIAMONDS_KING, None),     # Player 2: plays DIAMONDS_KING
        (3, ECard.SPADES_10, None),         # Player 3: plays SPADES_10
        (0, ECard.SPADES_9, None),          # Player 0: plays SPADES_9
    ]

    simulate_trick(pf, plays, my_hand=my_hand, rejuvenate_rate=0.05)


if __name__ == "__main__":
    main()
