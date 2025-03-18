import os
import argparse
import numpy as np
import torch as th
from typing import Optional

import robosuite as suite
from robosuite.wrappers import GymWrapper

from miga_sac import MIGA_SAC


def evaluate_policy(
    model: MIGA_SAC,
    env: GymWrapper,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True
) -> tuple[float, float]:
    """
    评估策略的性能
    """
    episode_rewards = []
    episode_lengths = []
    
    for i in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            if render:
                env.render()
            
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose:
            print(f"Episode {i+1}: reward = {episode_reward:.2f}, length = {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    if verbose:
        print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    
    return mean_reward, std_reward


def main(args):
    # 创建环境
    env = GymWrapper(
        suite.make(
            args.env,
            robots="Panda",
            use_camera_obs=args.use_camera_obs,
            has_offscreen_renderer=args.has_offscreen_renderer,
            has_renderer=args.has_renderer,
            reward_shaping=args.reward_shaping,
            control_freq=args.control_freq,
            horizon=500,
            reward_scale=1.0,
        )
    )

    # 加载模型
    model = MIGA_SAC.load(args.model_path, env=env)
    
    # 评估策略
    mean_reward, std_reward = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=not args.stochastic,
        render=args.render,
        verbose=args.verbose
    )
    
    # 保存评估结果
    if args.save_results:
        results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_eval_episodes": args.n_eval_episodes,
            "env_id": args.env,
            "model_path": args.model_path
        }
        
        os.makedirs("results", exist_ok=True)
        np.save(f"results/eval_results_{args.env}.npy", results)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIGA Policy Evaluation")
    
    # 评估参数
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the saved model")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions for evaluation")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--save-results", action="store_true",
                        help="Save evaluation results to a file")
    
    # 环境参数
    parser.add_argument("--env", type=str, default="Lift",
                        help="Robosuite environment name")
    parser.add_argument("--use-camera-obs", action="store_true",
                        help="Use camera observations")
    parser.add_argument("--has-offscreen-renderer", action="store_true",
                        help="Use offscreen renderer")
    parser.add_argument("--has-renderer", action="store_true",
                        help="Use renderer")
    parser.add_argument("--reward-shaping", action="store_true",
                        help="Use reward shaping")
    parser.add_argument("--control-freq", type=int, default=20,
                        help="Control frequency")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true",
                        help="Print evaluation progress")
    
    args = parser.parse_args()
    main(args)