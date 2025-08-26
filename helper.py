# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:18:55 2021

@author: LENOVO
"""

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training (CarGameAI)')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")

    # Đảm bảo trục y hiển thị từ 0 tới max
    y_max = max(max(scores), max(mean_scores)) + 10
    plt.ylim(0, y_max)

    # Hiển thị text điểm số cuối cùng
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1]+2, f"{scores[-1]}", color="blue")
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1]+2,
                 f"{round(mean_scores[-1],2)}", color="orange")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # hiển thị realtime và vẫn lưu ra file
    plt.savefig('Car_Game_AI.png')
    plt.pause(0.1)


def plot_eval(scores):
    """Vẽ biểu đồ đánh giá sau training"""
    plt.figure(figsize=(8,5))
    plt.title("Evaluation Results")
    plt.xlabel("Game")
    plt.ylabel("Score")

    plt.plot(scores, marker="o", linestyle="-", label="Eval Score")
    plt.axhline(y=np.mean(scores), color="orange", linestyle="--", label=f"Mean: {np.mean(scores):.2f}")
    plt.axhline(y=np.max(scores), color="green", linestyle="--", label=f"Max: {np.max(scores)}")
    plt.axhline(y=np.min(scores), color="red", linestyle="--", label=f"Min: {np.min(scores)}")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("Evaluation_Result.png")
    plt.show()