import random

class FullyRandomAgent:
    def __init__(self):
        pass

    def play(self, round):
        round.take_action(random.choice(list(round.get_actions())))