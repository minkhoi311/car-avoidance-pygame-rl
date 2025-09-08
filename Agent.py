import torch
import random
import numpy as np
from collections import deque
from Cargame_AI import CarGameAI, Direction
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # control randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # pop left
        self.model = Linear_QNet(6, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        sensor_1 = game.sensor_1
        sensor_2 = game.sensor_2
        sensor_3 = game.sensor_3

        state = [
            # car position (one-hot 3 lane)
            game.state == 0,
            game.state == 1,
            game.state == 2,
            # sensors
            game._get_sensor_response(sensor_1),
            game._get_sensor_response(sensor_2),
            game._get_sensor_response(sensor_3)
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def evaluate(agent, n_games=50, render=False):
    """Đánh giá agent sau khi train xong"""
    eval_scores = []
    game = CarGameAI(render=render, win_score=500)

    for i in range(n_games):
        game.reset()
        done = False
        total_score = 0

        while not done:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)  # chọn hành động
            reward, done, score = game.play_step(final_move)
            total_score = score

        eval_scores.append(total_score)
        print(f"[Eval] Game {i+1}/{n_games} | Score: {total_score}")

    mean_score = np.mean(eval_scores)
    max_score = np.max(eval_scores)
    min_score = np.min(eval_scores)

    print("\n=== Evaluation Report ===")
    print(f"Số game đánh giá : {n_games}")
    print(f"Điểm trung bình  : {mean_score:.2f}")
    print(f"Điểm cao nhất    : {max_score}")
    print(f"Điểm thấp nhất   : {min_score}")

    return mean_score, max_score, min_score


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()

    game = CarGameAI(render=False, win_score=500)

    while agent.n_games <= 500:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score  # chỉ cập nhật record, chưa lưu
            print('Game:', agent.n_games, ' Score:', score, ' Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

    # Sau khi train xong thì lưu model một lần
    agent.model.save("final_model.pth")
    print("\nTraining xong, đã lưu model -> final_model.pth")

if __name__ == '__main__':
    train()

