# -*- coding: utf-8 -*-
import torch
import numpy as np
from Cargame_AI import CarGameAI
from model import Linear_QNet

class PlayAgent:
    def __init__(self):
        self.model = Linear_QNet(6, 128, 3)
        # Load the trained model
        self.model.load_state_dict(torch.load('./model/model.pth'))
        self.model.eval()  # Set the model to evaluation mode

    def get_state(self, game):
        sensor_1 = game.sensor_1
        sensor_2 = game.sensor_2
        sensor_3 = game.sensor_3

        state = [
            # get car position
            game.state == 0,
            game.state == 1,
            game.state == 2,

            # get obs info
            game._get_sensor_response(sensor_1),
            game._get_sensor_response(sensor_2),
            game._get_sensor_response(sensor_3)
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move


def play():
    agent = PlayAgent()

    while True:   # Cho phép chơi lại nhiều lần
        game = CarGameAI()
        print("Starting game...")

        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            _, done, score = game.play_step(final_move)

            # Điều kiện thắng
            if score >= 100:
                print("🎉 Win! Né được 100 xe!")
                break

            if done:
                print(f"💀 Game Over! Final Score: {score}")
                break

        # Hỏi người chơi có muốn chơi lại không
        choice = input("Chơi lại? (y/n): ").strip().lower()
        if choice != "y":
            print("Thoát game.")
            break


if __name__ == '__main__':
    play()
