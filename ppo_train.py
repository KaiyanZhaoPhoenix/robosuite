import os
import warnings
import argparse
import time
import random
import numpy as np
import torch as th
import wandb
import json

from typing import Any, ClassVar, Optional, TypeVar, Union

from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.logger import configure

import robosuite as suite
from robosuite.wrappers import GymWrapper

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    [其余文档说明保持不变]
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,  # 优化后
        n_steps: int = 4096,  # 优化后
        batch_size: int = 64,
        n_epochs: int = 20,  # 优化后
        gamma: float = 0.99,
        gae_lambda: float = 0.98,  # 优化后
        clip_range: Union[float, Schedule] = 0.1,  # 优化后
        clip_range_vf: Union[None, float, Schedule] = 0.1,  # 优化后
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,  # 优化后
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = 0.01,  # 优化后
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 1,  # 优化后
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # 1. 设置自定义属性
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        # 2. 调用父类的 __init__，并设置 _init_setup_model=False
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=RolloutBuffer,
            rollout_buffer_kwargs=None,  # 避免传递冲突参数
            stats_window_size=100,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,  # 我们将在稍后手动调用 _setup_model
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # 3. 进行必要的检查
        if normalize_advantage:
            assert batch_size > 1, "`batch_size` must be greater than 1."

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {buffer_size // batch_size} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        # 4. 手动调用 _setup_model
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # 初始化 clip_range 和 clip_range_vf 的调度函数
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # 切换到训练模式（影响 BatchNorm/Dropout）
        self.policy.set_training_mode(True)
        # 更新优化器的学习率
        self._update_learning_rate(self.policy.optimizer)
        # 计算当前的剪切范围
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # 可选：值函数的剪切范围
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # 训练 n_epochs 轮
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # 完整遍历 rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # 将离散动作从 float 转为 long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # 归一化优势
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 计算比例
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # 裁剪的代理损失
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # 记录损失
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # 不裁剪
                    values_pred = values
                else:
                    # 裁剪值函数的预测
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # 使用 TD(gae_lambda) 目标计算值函数损失
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # 熵损失鼓励探索
                if entropy is None:
                    # 当没有解析形式时，近似熵
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # 计算近似反向 KL 散度，用于提前停止
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # 优化步骤
                self.policy.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # 记录日志
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


# 策略别名定义
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy


def set_seed(seed: int):
    th.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def initialize_wandb(args):
    return wandb.init(
        project="ppo_training",
        name=f"ppo_{args.env}_{args.seed}",
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
        notes="Training PPO on robosuite environment",
        mode='online'
    )


def main(args):
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    set_seed(args.seed)

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run = initialize_wandb(args)

    env = GymWrapper(
        suite.make(
            args.env,
            robots="Panda",
            use_camera_obs=args.use_camera_obs,
            has_offscreen_renderer=args.has_offscreen_renderer,
            has_renderer=args.has_renderer,
            reward_shaping=args.reward_shaping,
            control_freq=args.control_freq,
        )
    )
    env.reset(seed=args.seed)

    logger = configure(folder=args.log_folder, format_strings=["stdout", "csv", "tensorboard"])

    model = PPO(
        policy=args.policy,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        normalize_advantage=args.normalize_advantage,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        target_kl=args.target_kl,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=args.policy_kwargs,
        verbose=args.verbose,
        device=device,
    )
    model.set_logger(logger)

    model.learn(total_timesteps=args.total_timesteps)
    if args.save_model:
        model_path = os.path.join("models", f"ppo_{args.env}_{args.seed}")
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")

    run.finish()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO Training Script')

    # 环境参数
    parser.add_argument('--env', type=str, default="Stack",
                        help='Robosuite environment name')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--total_timesteps', type=int, default=2000000,  # 优化后
                        help='Total timesteps for training')

    # PPO 参数
    parser.add_argument('--policy', type=str, default="MlpPolicy",
                        choices=["MlpPolicy", "CnnPolicy", "MultiInputPolicy"],
                        help='Policy type')
    parser.add_argument('--learning_rate', type=float, default=1e-4,  # 优化后
                        help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=4096,  # 优化后
                        help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=20,  # 优化后
                        help='Number of epochs')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.98,  # 优化后
                        help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.1,  # 优化后
                        help='Clipping range')
    parser.add_argument('--clip_range_vf', type=float, default=0.1,  # 优化后
                        help='Clipping range for value function')
    parser.add_argument('--normalize_advantage', action='store_true',
                        help='Whether to normalize advantage')
    parser.add_argument('--ent_coef', type=float, default=0.01,  # 优化后
                        help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Max gradient norm')
    parser.add_argument('--use_sde', action='store_true',
                        help='Use state-dependent exploration')
    parser.add_argument('--sde_sample_freq', type=int, default=-1,
                        help='SDE sample frequency')
    parser.add_argument('--target_kl', type=float, default=0.01,  # 优化后
                        help='Target KL divergence')

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
