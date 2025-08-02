from Game import Game
from GameConfig import *
from DQNNetwork import DQNNetwork
import random
import numpy as np
import matplotlib.pyplot as plt

class TrainGame:
    def __init__(self, model_file=None):
        self.gamma = 0.5
        self.alpha = 0.01
        self.network = DQNNetwork(BOARD_WIDTH, BOARD_HEIGHT, len(SHIPS))
        if not model_file is None:
            self.network.restoreModel(r'C:\Users\baoph\Downloads\BattleShip\BattleShip\mymodel.keras')
        self.game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, network=self.network)
        self.total_ships_lengths = sum([ship['length'] for ship in self.game.board.ships])
        self.board_size = self.game.board.board_height * self.game.board.board_width
        self.max_train_step = 3000
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_moves = []

    def selfPlayOneGame(self):
        all_input_states = []
        all_moves = []
        all_hits = []
        self.game.resetBoard()
        (input_dimensions, move, is_hit) = self.game.takeAMove()
        while not input_dimensions is None:
            all_input_states.append(input_dimensions)
            all_moves.append(move)
            all_hits.append(is_hit)
            (input_dimensions, move, is_hit) = self.game.takeAMove()
        all_discounted_reward = self.rewardsCalculator(all_hits)
        return (all_input_states, all_moves, all_hits, all_discounted_reward)

    def rewardsCalculator(self, hit_log, gamma=0.5):
        hit_log_weighted = []
        for index, item in enumerate(hit_log):
            remaining_ships_length = self.total_ships_lengths - sum(hit_log[:index])
            remaining_board_size = self.board_size - index
            if remaining_board_size > 0:
                weighted_item = (item - float(remaining_ships_length) / float(remaining_board_size)) * (gamma ** index)
            else:
                weighted_item = item  
            hit_log_weighted.append(weighted_item)
        return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]

    def trainWithSelfPlay(self):
        batch_size = 50
        total_games = 0
        total_wins = 0
        total_moves = 0
        for i in range(self.max_train_step):
            (all_input_states, all_moves, all_hits, all_discounted_reward) = self.selfPlayOneGame()
            total_games += 1
            total_hits = sum(all_hits)
            total_moves += len(all_hits)
            if total_hits > 0:
                total_wins += 1
            self.episode_rewards.append(sum(all_discounted_reward))
            self.episode_wins.append(total_wins)
            self.episode_moves.append(total_moves)
            for input_states, moves, discounted_reward in zip(all_input_states, all_moves, all_discounted_reward):
                entropy = self.network.runTrainStep(input_states, [moves], self.alpha * discounted_reward)
            self.network.decay_epsilon()
            if i % batch_size == 0 and i != 0:
                avg_moves = total_moves / batch_size
                win_rate = total_wins / batch_size
                print(f"Episode: {i}/{self.max_train_step}")
                print(f"Epsilon: {self.network.epsilon:.4f}")
                print(f"Average Moves: {avg_moves}")
                print(f"Win Rate: {win_rate:.4f}")
                print(f"Total Reward: {sum(all_discounted_reward)}")
                print("-" * 50)
                total_wins = 0
                total_moves = 0
                total_games = 0
            if i % (batch_size * 20) == 0:
                self.game.network.saveModel('./models/mymodel')
                print(f"Model saved at step {i}")
        self.plot_training_results()

    def plot_training_results(self):
        episodes = range(len(self.episode_rewards))
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(episodes, self.episode_rewards, label='Total Reward', color='blue')
        plt.title('Total Reward vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(episodes, self.episode_wins, label='Wins', color='green')
        plt.title('Wins vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Wins')
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(episodes, self.episode_moves, label='Moves', color='red')
        plt.title('Moves vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Moves')
        plt.grid()
        plt.tight_layout()
        plt.show()

train_game = TrainGame('./models/mymodel')
train_game.trainWithSelfPlay()