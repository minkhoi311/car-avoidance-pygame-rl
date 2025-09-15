import matplotlib
matplotlib.use("Agg")   # dùng backend không cần GUI
import matplotlib.pyplot as plt
import numpy as np

def plot(scores, mean_scores, save_path="Training_Result.png"):
    plt.figure(figsize=(8,5))
    plt.title('Training (CarGameAI)')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label="Score", color="blue")
    plt.plot(mean_scores, label="Mean Score", color="orange")

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
    plt.savefig(save_path)
    plt.close()  # đóng figure sau khi lưu

def plot_eval(scores, save_path="Evaluation_Result.png"):
    """Vẽ biểu đồ đánh giá sau training"""
    plt.figure(figsize=(8,5))
    plt.title("Evaluation Results")
    plt.xlabel("Game")
    plt.ylabel("Score")

    plt.plot(scores, marker="o", linestyle="-", label="Eval Score", color="blue")
    plt.axhline(y=np.mean(scores), color="orange", linestyle="--", label=f"Mean: {np.mean(scores):.2f}")
    plt.axhline(y=np.max(scores), color="green", linestyle="--", label=f"Max: {np.max(scores)}")
    plt.axhline(y=np.min(scores), color="red", linestyle="--", label=f"Min: {np.min(scores)}")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path)
    plt.close()  # đóng figure sau khi lưu
