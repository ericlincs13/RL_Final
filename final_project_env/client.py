import os
import argparse
import json
import numpy as np
import gymnasium as gym


def get_observation(url=None):
    if not url:
        import server

        data = server.get_observation()
    else:
        import requests

        response = requests.get(f"{url}")
        data = json.loads(response.text)

    if data.get("error"):
        raise RuntimeError(data["error"])
    return data


def set_action(action, url=None, skip_print=False):
    if not url:
        import server

        data = server.set_action(action, skip_print=skip_print)
    else:
        import requests

        response = requests.post(f"{url}", json={"action": action.tolist()})
        data = json.loads(response.text)

    if data.get("error"):
        raise RuntimeError(data["error"])
    return data


class RemoteRacecarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, url=None, action_low=None, action_high=None):
        self.url = url

        if not url:
            import server

            server.init_server()

        # Probe one observation to infer shape/dtype
        first = get_observation(self.url)
        obs = np.asarray(first["observation"], dtype=np.uint8)

        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs.shape, dtype=np.uint8
        )
        # Default to 2D continuous actions in [-1, 1] as used by the client agent
        low = (
            np.array([-1, -1], dtype=np.float32)
            if action_low is None
            else np.array(action_low, dtype=np.float32)
        )
        high = (
            np.array([1, 1], dtype=np.float32)
            if action_high is None
            else np.array(action_high, dtype=np.float32)
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Keep last obs to serve as initial obs after reset()
        self._last_obs = obs

    # Gymnasium API
    def reset(self, *, seed=None, options=None):
        # Request a fresh observation; server is expected to handle episode resets
        data = get_observation(self.url)
        obs = np.asarray(data["observation"], dtype=np.uint8)
        self._last_obs = obs
        info = {}
        return obs, info

    def step(self, action):
        # Ensure action is within bounds and convert to list for JSON
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        data = set_action(action, self.url, skip_print=True)
        reward = float(data.get("reward", 0.0))
        terminated = bool(data.get("terminal", False))
        truncated = bool(data.get("truncated", False))
        info = {}

        data = get_observation(self.url)
        obs = np.asarray(data["observation"], dtype=np.uint8)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass


class CarRacingAgent:
    def __init__(self, action_space, model_path=None, deterministic=True, device="cpu"):
        self.action_space = action_space
        self.deterministic = deterministic
        self.model = None
        # Lazy import of SB3 and attempt to load PPO model (no env creation)
        try:
            from stable_baselines3 import PPO  # type: ignore
            import os

            if model_path is not None and os.path.isfile(model_path):
                self.model = PPO.load(model_path, device=device)
                print(f"Loaded PPO model from: {model_path}")
            else:
                if model_path:
                    print(
                        f"PPO model not found at: {model_path} — falling back to random actions."
                    )
        except Exception as e:
            # SB3 not installed or load failed
            print(
                f"PPO not available or failed to load ({e}) — falling back to random actions."
            )

    def act(self, observation):
        # Prefer PPO policy inference when available
        if self.model is not None:
            try:
                action, _ = self.model.predict(
                    observation, deterministic=self.deterministic
                )
                action = np.asarray(action, dtype=np.float32)
                # Safety: clip to declared action bounds
                action = np.clip(action, self.action_space.low, self.action_space.high)
                return action
            except Exception as e:
                print(f"PPO predict failed ({e}) — falling back to random action.")
        # Fallback: random continuous action
        return self.action_space.sample()


def connect(agent, url=None):
    while True:
        # Get the observation
        data = get_observation(url)
        if data.get("error"):
            print(data["error"])
            break
        obs = np.array(data["observation"]).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        data = set_action(action_to_take, url)
        if data.get("error"):
            print(data["error"])
            break
        terminal = data["terminal"]

        if terminal:
            print("Episode finished.")
            return


def training(args):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as e:
        raise RuntimeError(f"Stable-Baselines3 is required for training: {e}")

    # Prefer GPU when requested
    device = args.device
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU.")
                device = "cpu"
        except Exception:
            print(
                "Warning: torch not available to check CUDA; continuing with 'cuda' which may fail."
            )

    def _make_env():
        return lambda: RemoteRacecarEnv(url=args.url)

    # PPO for continuous control
    env = SubprocVecEnv([_make_env() for _ in range(args.n_envs)])
    model = PPO(
        policy="CnnPolicy",
        env=env,
        device=device,
        verbose=0,
    )

    print(
        f"Start PPO training on device={device} for total_timesteps={args.total_timesteps}"
    )
    # Prepare checkpoint and (optional) evaluation callbacks for periodic saving/eval
    os.makedirs(args.save_dir, exist_ok=True)
    callback_list = []

    checkpoint_callback = CheckpointCallback(
        save_freq=int(args.save_freq),
        save_path=args.save_dir,
        name_prefix="ppo_racecar",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callback_list.append(checkpoint_callback)

    # Optional: periodic evaluation during training
    if args.eval_freq > 0:
        os.makedirs(args.eval_log_dir, exist_ok=True)
        eval_env = RemoteRacecarEnv(url=args.url)
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=int(args.eval_episodes),
            eval_freq=int(args.eval_freq),
            log_path=args.eval_log_dir,
            deterministic=True,
            render=False,
        )
        callback_list.append(eval_callback)

    # Use a single callback or a CallbackList depending on how many we have
    if len(callback_list) == 1:
        callback = callback_list[0]
    else:
        callback = CallbackList(callback_list)

    model.learn(total_timesteps=int(args.total_timesteps), callback=callback)
    model.save(f"{args.save_dir}/ppo_racecar_final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        help="URL of the remote server.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to SB3 PPO model (.zip). If not found, fallback to random actions.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the PPO agent against the remote server.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1e8,
        help="Total timesteps for PPO training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use for PPO ("cuda" for GPU, or "cpu").',
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N environment steps during training.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="weights",
        help="Directory to store periodic PPO checkpoints.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel remote envs.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help=(
            "Evaluate the PPO agent every N environment steps during training "
            "(set <= 0 to disable evaluation)."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run each time evaluation is triggered.",
    )
    parser.add_argument(
        "--eval-log-dir",
        type=str,
        default="eval_logs",
        help="Directory to store evaluation logs (episode rewards, lengths, etc.).",
    )
    args = parser.parse_args()

    # Train or run inference
    if not args.eval:
        training(args)
    else:
        agent = CarRacingAgent(
            action_space=gym.spaces.Box(
                low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
            ),
            model_path=args.model,
            deterministic=True,
            device=args.device,
        )

        if not args.url:
            import server

            server.init_server()

        connect(agent, url=args.url)
