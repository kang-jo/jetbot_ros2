import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# CONFIG
# =========================
# CSV_PATH = ""
CSV_PATH = r"training_log_HITL_20260521.csv"
SUCCESS_WINDOW = 21
MA_WINDOW = 14

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

episodes = df["episode"].to_numpy()
rewards = df["total_reward"].to_numpy()
steps = df["steps"].to_numpy()
hitl = df["hitl_overrides"].to_numpy()
success = df["success"].to_numpy()

# moving success rate
success_series = pd.Series(success)
success_rate = success_series.rolling(
    window=SUCCESS_WINDOW, min_periods=1
).mean().to_numpy()

# =========================
# MOVING AVERAGE & AVERAGE
# =========================
reward_ma5 = pd.Series(rewards).rolling(window=MA_WINDOW, min_periods=1).mean().to_numpy()
steps_ma5 = pd.Series(steps).rolling(window=MA_WINDOW, min_periods=1).mean().to_numpy()

reward_avg = np.full_like(rewards, np.mean(rewards))
steps_avg = np.full_like(steps, np.mean(steps))

# =========================
# PLOT SEMUA DALAM 1 FIGURE
# =========================
# plt.style.use("seaborn-v0_8")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()

# ---- Reward ----
ax[0].plot(episodes, rewards, color="#0066FF", alpha=0.4, linewidth=1.5, label="Reward")
ax[0].plot(episodes, reward_ma5, color="#FF6600", linewidth=2.5, label="MA14")
ax[0].plot(episodes, reward_avg, color="#00AA55", linestyle="--", linewidth=2, label="Average")
ax[0].set_title("Cumulative Reward vs Episode")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Reward")
ax[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax[0].legend(loc="upper left")

# ---- Steps ----
ax[1].plot(episodes, steps, color="#0066FF", alpha=0.4, linewidth=1.5)
ax[1].plot(episodes, steps_ma5, color="#FF6600", linewidth=2.5)
ax[1].plot(episodes, steps_avg, color="#00AA55", linestyle="--", linewidth=2)
ax[1].set_title("Steps per Episode")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Steps")
ax[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

# ---- HITL ----
ax[2].plot(episodes, hitl, color="#9933FF", linewidth=2)
ax[2].set_title("HITL Overrides")
ax[2].set_xlabel("Episode")
ax[2].set_ylabel("Overrides")
ax[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

# ---- Success Rate ----
ax[3].plot(episodes, success_rate, color="#FF0033", linewidth=2)
ax[3].set_title(f"Success Rate")
ax[3].set_xlabel("Episode")
ax[3].set_ylabel("Success Rate")
ax[3].set_ylim(0, 1.05)
ax[3].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.show()