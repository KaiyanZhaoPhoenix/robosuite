import os
import warnings
import argparse
import time
import random
import numpy as np
import torch as th
import wandb
import json

from typing import Any, Optional, Union

from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

import robosuite as suite
from robosuite.wrappers import GymWrapper


def set_seed(seed: int):
    """
    设置随机种子，确保实验可复现。
    """
    th.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def initialize_wandb(args):
    """
    初始化 Weights & Biases (WandB) 以记录训练过程中的指标和参数。
    """
    return wandb.init(
        project="sac_training",
        entity="rlma",
        name=f"sac_{args.env}_{args.seed}",
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
        notes="Training SAC on Robosuite environment",
        mode='online' if not args.debug else 'disabled'  # 可以通过命令行参数控制是否启用WandB
    )


def main(args):
    """
    主训练函数。
    """
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    set_seed(args.seed)

    #cuda
    # 修复 CUDA 设备选择逻辑
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    # 添加 GPU 选择功能
    if device == "cuda" and args.gpu_id is not None:
        device = f"cuda:{args.gpu_id}"
    
    print(f"Using device: {device}")

    run = initialize_wandb(args)

    # 创建Robosuite环境并用GymWrapper包装
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
    env.reset(seed=args.seed)

    # 配置日志记录器
    logger = configure(folder=args.log_folder, format_strings=["stdout", "csv", "tensorboard"])

    # 计算目标熵
    if args.policy == "MultiInputPolicy":
        action_space_dim = get_action_dim(env.action_space)
        target_entropy = -action_space_dim
    elif args.policy in ["CnnPolicy", "MlpPolicy"]:
        action_space_dim = get_action_dim(env.action_space)
        target_entropy = -action_space_dim
    else:
        target_entropy = -1.0  # 默认值

    # 初始化SAC模型
    model = SAC(
        policy=args.policy,
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=int(args.buffer_size),
        learning_starts=int(args.learning_starts),
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        target_entropy=target_entropy if args.target_entropy == "auto" else args.target_entropy,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=args.policy_kwargs,
        verbose=args.verbose,
        device=device,
        seed=args.seed,
    )
    model.set_logger(logger)

    # 开始训练
    model.learn(total_timesteps=args.total_timesteps)
    
    # 保存模型
    if args.save_model:
        model_path = os.path.join("models", f"sac_{args.env}_{args.seed}")
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")

    run.finish()
    env.close()


def get_action_dim(action_space: spaces.Space) -> int:
    """
    获取动作空间的维度。
    """
    if isinstance(action_space, spaces.Box):
        return action_space.shape[0]
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(np.prod(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        return action_space.n
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC Training Script')

    # 环境参数
    parser.add_argument('--env', type=str, default="Lift",
                        help='Robosuite environment name')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--total_timesteps', type=int, default=1e10,  # 优化后
                        help='Total timesteps for training')

    # SAC 参数
    parser.add_argument('--policy', type=str, default="MlpPolicy",
                        choices=["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
                        help='Policy type')
    parser.add_argument('--learning_rate', type=float, default=3e-4,  # 优化后
                        help='Learning rate')
    parser.add_argument('--buffer_size', type=float, default=1_000_000,  # 优化后
                        help='Replay buffer size')
    parser.add_argument('--learning_starts', type=float, default=10_000,  # 优化后
                        help='Number of steps to collect transitions before learning starts')
    parser.add_argument('--batch_size', type=int, default=256,  # 优化后
                        help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,  # 优化后
                        help='Soft update coefficient')
    parser.add_argument('--ent_coef', type=str, default="auto",  # 优化后
                        help='Entropy coefficient ("auto" or float)')
    parser.add_argument('--target_entropy', type=str, default="auto",  # 新增
                        help='Target entropy ("auto" or float)')
    parser.add_argument('--train_freq', type=int, default=1,  # 优化后
                        help='Number of steps between updates')
    parser.add_argument('--gradient_steps', type=int, default=1,  # 优化后
                        help='Number of gradient steps after each rollout')
    parser.add_argument('--target_kl', type=float, default=None,  # 可选参数
                        help='Target KL divergence (not typically used in SAC)')

    # 环境配置
    parser.add_argument('--use_camera_obs', action='store_true',
                        help='Use camera observations')
    parser.add_argument('--has_offscreen_renderer', action='store_true',
                        help='Use offscreen renderer')
    parser.add_argument('--has_renderer', action='store_true', default=False,
                        help='Use renderer')
    parser.add_argument('--reward_shaping', action='store_true', default=False,
                        help='Use reward shaping')
    parser.add_argument('--control_freq', type=int, default=20,
                        help='Control frequency')

    # 日志
    parser.add_argument('--log_folder', type=str, default="./logs/",
                        help='Logging folder')
    parser.add_argument('--tensorboard_log', type=str, default="./tensorboard/",
                        help='TensorBoard log folder')
    parser.add_argument('--policy_kwargs', type=str, default=None,
                        help='Additional policy keyword arguments as JSON string')
    parser.add_argument('--verbose', type=int, default=1,  # 优化后
                        help='Verbosity level')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model after training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (disable WandB logging)')

    # 在环境配置部分添加
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='指定使用的 GPU ID (如果有多个 GPU)')

    args = parser.parse_args()

    # 解析 policy_kwargs
    if args.policy_kwargs is not None:
        try:
            args.policy_kwargs = json.loads(args.policy_kwargs)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for --policy_kwargs")

    training_start_time = time.time()
    main(args)
    training_duration = time.time() - training_start_time
    print(f'Training time: {training_duration / 3600:.2f} hours')