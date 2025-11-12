import random

from euchre.game import Game
from euchre.text_interface import text_interface
from euchre.round import PLAYING_STATE

# Play a game with random legal moves
# (Players 0 and 2 are on team 0 and players 1 and 3 are on team 1)
game = Game()

while not game.finished:
    text_interface(game)

    actions = game.get_actions()
    choice = random.choice(list(actions))

    print(f"\n> Playing action {choice}!")

    old_state = game.round.estate
    old_team_points = game.team_points.copy()

    game.take_action(choice)

    if old_state != game.round.estate and old_state == PLAYING_STATE:
        print(f"\n> Round finished! Team 0 got {game.team_points[0] - old_team_points[0]} points and team 1 got {game.team_points[1] - old_team_points[1]} points!")

print(f"\n\nThe winning team is team {game.winner}! The scores were team 0 with {game.team_points[0]} points and team 1 with {game.team_points[1]} points!")