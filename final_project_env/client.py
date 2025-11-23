import argparse
import json
import numpy as np
import requests
import gymnasium as gym


class RemoteRacecarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, url: str, action_low=None, action_high=None):
        self.url = url
        # Probe one observation to infer shape/dtype
        first = requests.get(f"{self.url}")
        if json.loads(first.text).get("error"):
            raise RuntimeError(json.loads(first.text)["error"])
        obs = np.asarray(json.loads(first.text)["observation"], dtype=np.uint8)

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
        response = requests.get(f"{self.url}")
        if json.loads(response.text).get("error"):
            raise RuntimeError(json.loads(response.text)["error"])
        data = json.loads(response.text)
        obs = np.asarray(data["observation"], dtype=np.uint8)
        self._last_obs = obs
        info = {}
        return obs, info

    def step(self, action):
        # Ensure action is within bounds and convert to list for JSON
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        response = requests.post(f"{self.url}", json={"action": action.tolist()})
        if json.loads(response.text).get("error"):
            raise RuntimeError(json.loads(response.text)["error"])
        data = json.loads(response.text)
        reward = float(data.get("reward", 0.0))
        terminated = bool(data.get("terminal", False))
        # Server POST may not include observation; fetch it if missing
        if "observation" in data:
            obs = np.asarray(data["observation"], dtype=np.uint8)
        else:
            get_resp = requests.get(f"{self.url}")
            if json.loads(get_resp.text).get("error"):
                raise RuntimeError(json.loads(get_resp.text)["error"])
            get_data = json.loads(get_resp.text)
            obs = np.asarray(get_data["observation"], dtype=np.uint8)
        # No explicit truncation signal in the protocol; default to False
        truncated = False
        info = {}
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


def connect(agent, url: str = "http://localhost:5000"):
    while True:
        # Get the observation
        response = requests.get(f"{url}")
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break
        obs = json.loads(response.text)["observation"]
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f"{url}", json={"action": action_to_take.tolist()})
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break

        result = json.loads(response.text)
        terminal = result["terminal"]

        if terminal:
            print("Episode finished.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for PPO.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for PPO.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100000,
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
        help="Number of parallel remote envs (e.g., 8 for ports 5001-5008).",
    )
    parser.add_argument(
        "--port-start",
        type=int,
        default=5001,
        help="Starting port for parallel envs (inclusive).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost",
        help="Remote host scheme+domain used to build env URLs (e.g., http://localhost).",
    )
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="subproc",
        choices=["subproc", "dummy"],
        help="Vector env backend to use for parallelism.",
    )
    args = parser.parse_args()

    # Train or run inference
    if not args.eval:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback
            from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
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

        # Create vectorized remote proxy envs (no local physics)
        # Build URLs like http://localhost:5001 ... :5001 + n_envs - 1
        urls = [f"{args.host}:{args.port_start + i}" for i in range(int(args.n_envs))]

        def _make_env(_url):
            return lambda: RemoteRacecarEnv(_url)

        env_fns = [_make_env(u) for u in urls]
        if args.vec_backend == "subproc":
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)

        # Choose policy: default CnnPolicy for image observation
        policy = "CnnPolicy"

        # Enable gSDE for continuous control per SB3 recommendation notes
        model = PPO(
            policy,
            env,
            device=device,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            verbose=1,
            use_sde=True,
        )

        print(
            f"Start PPO training on device={device} for total_timesteps={args.total_timesteps}"
        )
        # Prepare checkpoint callback for periodic saving
        import os

        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=int(args.save_freq),
            save_path=args.save_dir,
            name_prefix="ppo_racecar",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model.learn(
            total_timesteps=int(args.total_timesteps), callback=checkpoint_callback
        )
        model.save(args.model)
        print(f"Saved PPO model to: {args.model}")
    else:
        # Initialize the RL Agent for inference
        import gymnasium as gym

        rand_agent = CarRacingAgent(
            action_space=gym.spaces.Box(
                low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
            ),
            model_path=args.model,
            deterministic=True,
            device=args.device,
        )

        connect(rand_agent, url=args.url)
