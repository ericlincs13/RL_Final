import argparse
import json
import numpy as np
import requests


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
        "--url",
        type=str,
        default="http://localhost:5000",
        help="The url of the server.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_racecar.zip",
        help="Path to SB3 PPO model (.zip). If not found, fallback to random actions.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions at inference (recommended for continuous control).",
    )
    args = parser.parse_args()

    class RandomAgent:
        def __init__(self, action_space, model_path=None, deterministic=True):
            self.action_space = action_space
            self.deterministic = deterministic
            self.model = None
            # Lazy import of SB3 and attempt to load PPO model (no env creation)
            try:
                from stable_baselines3 import PPO  # type: ignore
                import os

                if model_path is not None and os.path.isfile(model_path):
                    self.model = PPO.load(model_path, device="cpu")
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
                    action = np.clip(
                        action, self.action_space.low, self.action_space.high
                    )
                    return action
                except Exception as e:
                    print(f"PPO predict failed ({e}) — falling back to random action.")
            # Fallback: random continuous action
            return self.action_space.sample()

    # Initialize the RL Agent
    import gymnasium as gym

    rand_agent = RandomAgent(
        action_space=gym.spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
        ),
        model_path=args.model,
        deterministic=bool(args.deterministic),
    )

    connect(rand_agent, url=args.url)
