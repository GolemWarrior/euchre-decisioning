from enum import IntEnum

from .round import Round

class Game:
    def __init__(self):
        self.round = Round()
        self.team_points = [0] * 2

        self.finished = False
        self.winner = None

    def get_actions(self) -> set[IntEnum]:
        return self.round.get_actions()

    def take_action(self, action: IntEnum):
        self.round.take_action(action)

        if self.round.finished:
            for i in range(2):
                self.team_points[i] += self.round.round_points[i]

            # First to ten points wins
            most_points = max(self.team_points)
            if most_points >= 10:
                self.finished = True
                self.winner = self.team_points.index(most_points)  # Only one team can get points per round, so can't tie first to ten

            self.round = Round(last_dealer=self.round.dealer)

    def get_current_player(self):
        return self.round.get_current_player()