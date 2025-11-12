from .round import FIRST_BIDDING_STATE, DEALER_DISCARD_STATE, SECOND_BIDDING_STATE, CHOOSING_ESUIT_STATE, DECIDING_GOING_ALONE_STATE, PLAYING_STATE

def text_interface(game):
    if game.finished:
        print("The game is finised!")
        return

    cur_round = game.round
    current_player = cur_round.current_player
    dealer = cur_round.dealer
    hand = cur_round.hands[current_player]
    hand_text = ", ".join([ecard.name if ecard is not None else "---" for ecard in hand])

    actions_text = "\n> Your options are:"
    for action in sorted(list(game.get_actions())):
        actions_text += f"\n\t> {action.value}: {action.name}"

    print(f"\n--- CURRENT PLAYER: \"{current_player.name}\" ({cur_round.estate.name}) ---\n")

    if cur_round.estate == FIRST_BIDDING_STATE:
        print("> It's the first round of bidding!")
        print(f"\t> Upcard: {cur_round.upcard.name}")
        print(f"\t> Your hand: {hand_text}")
        print(f"\t> Dealer: Player {cur_round.dealer}")

    elif cur_round.estate == DEALER_DISCARD_STATE:
        print("> Someone ordered up, so you (as the dealer) can discard and get the upcard!")
        print(f"\t> Upcard: {cur_round.upcard.name}")
        print(f"\t> Your hand: {hand_text}")

    elif cur_round.estate == SECOND_BIDDING_STATE:
        print("> It's the second round of bidding! Order up to pick any suit!")
        print(f"\t> Upcard: {cur_round.upcard.name}")
        print(f"\t> Your hand: {hand_text}")
        print(f"\t> Dealer: Player {cur_round.dealer}")

    elif cur_round.estate == CHOOSING_ESUIT_STATE:
        print("> You ordered up during the second bidding round! Pick any suit to be the trump!")
        print(f"\t> Your hand: {hand_text}")

    elif cur_round.estate == DECIDING_GOING_ALONE_STATE:
        print("> You ordered up! Decide whether to go alone!")
        print(f"\t> Your hand: {hand_text}")

    elif cur_round.estate == PLAYING_STATE:
        played_cards_text = ", ".join([ecard.name for ecard in cur_round.played_ecards])
        print(f"\t> Played cards: {played_cards_text}")
        print(f"\t> Your hand: {hand_text}")

    print(actions_text)