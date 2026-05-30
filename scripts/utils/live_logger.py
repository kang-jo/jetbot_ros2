import os
import csv
import threading
from datetime import datetime
from collections import deque


class LivePlotLogger:

    HEADER = ["episode", "total_reward", "steps", "hitl_overrides", "success", "epsilon"]

    def __init__(self, enable_plot=False, window=20, resume=True,
                 csv_episode_path=None, log_dir=None,
                 filename_prefix="training_log_HITL"):
        if enable_plot:
            print("[LOGGER] enable_plot=True ignored in safe logger. CSV logging remains active.")

        self.enable_plot = False
        self.window = int(window)
        self.resume = bool(resume)

        self.episode_offset = 0
        self.last_episode = 0
        self.last_epsilon = None

        self._lock = threading.RLock()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d")

        if log_dir is None:
            self.log_dir = os.path.normpath(os.path.join(base_dir, "..", "training_logs"))
        else:
            self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        if csv_episode_path is None:
            self.csv_episode_path = os.path.join(self.log_dir, f"{filename_prefix}_{timestamp}.csv")
        else:
            self.csv_episode_path = os.path.abspath(csv_episode_path)
            parent = os.path.dirname(self.csv_episode_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        self.episodes = []
        self.rewards = []
        self.steps = []
        self.hitl = []
        self.success = []
        self.epsilon = []
        self.success_window = deque(maxlen=self.window)

        if os.path.exists(self.csv_episode_path) and self.resume:
            self._load_existing_log()

        self._ensure_csv_header()

        if self.episode_offset > 0:
            print(
                f"[LOGGER] Resuming from episode {self.episode_offset} | "
                f"last_epsilon={self.last_epsilon if self.last_epsilon is not None else 'None'} | "
                f"SuccessRate{self.window}: {self.get_success_rate():.2f}"
            )
        else:
            print(f"[LOGGER] Starting new log: {self.csv_episode_path}")

    def _ensure_csv_header(self):
        if not os.path.exists(self.csv_episode_path) or os.path.getsize(self.csv_episode_path) == 0:
            with open(self.csv_episode_path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)
            return

        try:
            with open(self.csv_episode_path, "r", newline="") as f:
                first = f.readline().strip()
            expected = ",".join(self.HEADER)
            if first != expected:
                print(
                    f"[LOGGER] WARNING: CSV header differs from expected. "
                    f"Expected={self.HEADER}, path={self.csv_episode_path}"
                )
        except Exception as exc:
            print(f"[LOGGER] WARNING: failed to inspect CSV header: {exc}")

    def _parse_row(self, row):
        try:
            episode = int(float(row.get("episode", "")))
            total_reward = float(row.get("total_reward", ""))
            steps = int(float(row.get("steps", "")))
            hitl_overrides = int(float(row.get("hitl_overrides", 0)))
            success = 1 if int(float(row.get("success", 0))) else 0
            epsilon = float(row.get("epsilon", ""))
            if episode <= 0:
                return None
            return {
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "hitl_overrides": hitl_overrides,
                "success": success,
                "epsilon": epsilon,
            }
        except Exception:
            return None

    def _load_existing_log(self):
        rows = []
        try:
            with open(self.csv_episode_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = self._parse_row(row)
                    if parsed is not None:
                        rows.append(parsed)
        except Exception as exc:
            print(f"[LOGGER] WARNING: failed to load existing log: {exc}")
            return

        if not rows:
            return

        rows.sort(key=lambda x: x["episode"])

        with self._lock:
            self.episode_offset = int(rows[-1]["episode"])
            self.last_episode = self.episode_offset
            self.last_epsilon = float(rows[-1]["epsilon"])

            self.episodes = [int(r["episode"]) for r in rows]
            self.rewards = [float(r["total_reward"]) for r in rows]
            self.steps = [int(r["steps"]) for r in rows]
            self.hitl = [int(r["hitl_overrides"]) for r in rows]
            self.success = [int(r["success"]) for r in rows]
            self.epsilon = [float(r["epsilon"]) for r in rows]

            self.success_window.clear()
            for r in rows[-self.window:]:
                self.success_window.append(int(r["success"]))

    def get_success_rate(self):
        with self._lock:
            if not self.success_window:
                return 0.0
            return float(sum(self.success_window)) / float(len(self.success_window))

    def get_resume_state(self):
        with self._lock:
            success_rate = 0.0
            if self.success_window:
                success_rate = float(sum(self.success_window)) / float(len(self.success_window))
            return {
                "has_history": self.last_episode > 0,
                "last_episode": int(self.last_episode),
                "episode_offset": int(self.episode_offset),
                "last_epsilon": None if self.last_epsilon is None else float(self.last_epsilon),
                "success_rate": success_rate,
                "success_window_size": int(self.window),
                "csv_episode_path": self.csv_episode_path,
            }

    def log_episode(self, episode, total_reward, steps, hitl_count, success, epsilon,
                    absolute_episode=False):
        if absolute_episode:
            episode_real = int(episode)
        else:
            episode_real = int(episode) + int(self.episode_offset)

        total_reward = float(total_reward)
        steps = int(steps)
        hitl_count = int(hitl_count)
        success_int = 1 if success else 0
        epsilon = float(epsilon)

        with self._lock:
            self.episodes.append(episode_real)
            self.rewards.append(total_reward)
            self.steps.append(steps)
            self.hitl.append(hitl_count)
            self.success.append(success_int)
            self.epsilon.append(epsilon)
            self.success_window.append(success_int)
            self.last_episode = episode_real
            self.last_epsilon = epsilon
            success_rate = float(sum(self.success_window)) / float(len(self.success_window))

        with open(self.csv_episode_path, "a", newline="") as f:
            csv.writer(f).writerow([episode_real, total_reward, steps, hitl_count, success_int, epsilon])

        print(
            f"[LOGGER] Ep {episode_real} | Reward: {total_reward:.2f} | Steps: {steps} | "
            f"HITL: {hitl_count} | ε: {epsilon:.3f} | SuccessRate{self.window}: {success_rate:.2f}"
        )

    def close(self):
        print(f"[LOGGER] Episode log saved to: {self.csv_episode_path}")