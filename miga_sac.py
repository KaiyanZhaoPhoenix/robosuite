import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class DiscriminatorNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        x = th.cat([obs, actions], dim=-1)
        return self.net(x)


class MIGAReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_quality = np.zeros((self.buffer_size,), dtype=np.float32)
        self.current_success_rate = 0.0
        # 添加高效样本计数器
        self.high_quality_count = 0
    
    def add(self, *args, is_success: bool = False, **kwargs):
        pos = self.pos
        super().add(*args, **kwargs)
        quality = 1.0 if is_success else 0.0
        self.sample_quality[pos] = quality
        
        # 更新高效样本计数
        if self.full:
            if quality > 0.5 and self.sample_quality[pos] <= 0.5:
                self.high_quality_count += 1
            elif quality <= 0.5 and self.sample_quality[pos] > 0.5:
                self.high_quality_count -= 1
        else:
            if quality > 0.5:
                self.high_quality_count += 1
        
        # 更高效地计算成功率
        if self.full:
            self.current_success_rate = self.high_quality_count / self.buffer_size
        else:
            self.current_success_rate = self.high_quality_count / self.pos if self.pos > 0 else 0.0
    
    def sample_split(self, batch_size: int) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor]]:
        # 直接从缓冲区采样原始数据
        upper_bound = self.buffer_size if self.full else self.pos
        batch_indices = np.random.randint(0, upper_bound, size=batch_size)
        
        # 获取这些索引对应的质量值
        qualities = th.tensor(self.sample_quality[batch_indices], device=self.device)
        
        # 分离高质量和低质量样本
        high_mask = qualities > 0.5
        low_mask = ~high_mask
        
        # 获取对应的样本数据
        high_indices = batch_indices[high_mask.cpu().numpy()]
        low_indices = batch_indices[low_mask.cpu().numpy()]
        
        # 如果任一组为空，返回空字典
        if len(high_indices) == 0 or len(low_indices) == 0:
            return {}, {}
        
        # 为高质量样本创建批次
        high_batch = self._get_samples(high_indices)
        # 为低质量样本创建批次
        low_batch = self._get_samples(low_indices)
        
        # 转换为字典格式
        high_batch_dict = {k: v for k, v in zip(self.field_names, high_batch)}
        low_batch_dict = {k: v for k, v in zip(self.field_names, low_batch)}
        
        return high_batch_dict, low_batch_dict
    
    def sample_negative_actions(self, obs: th.Tensor, batch_size: int = 10) -> th.Tensor:
        # 直接从缓冲区采样原始数据
        upper_bound = self.buffer_size if self.full else self.pos
        batch_indices = np.random.randint(0, upper_bound, size=batch_size)
        
        # 获取这些索引对应的动作
        actions = self._get_samples(batch_indices)[1]  # 假设动作是第二个返回值
        
        return actions


class MIGA_SAC(SAC):
    def __init__(
        self,
        policy: str,
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[Any] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        beta_init: float = 0.5,
        lambda_decay: float = 0.001,
    ):
        if replay_buffer_class is None:
            replay_buffer_class = MIGAReplayBuffer
        
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )
        
        self.beta = beta_init
        self.lambda_decay = lambda_decay
        self.n_updates = 0
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """设置模型组件"""
        super()._setup_model()
        
        # 创建低效Critic网络
        self.critic_low = self.critic
        self.critic_low_target = self.critic_target
        
        # 创建互信息判别器
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        self.discriminator = DiscriminatorNetwork(obs_dim, action_dim).to(self.device)
        self.discriminator_optimizer = th.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        # 创建固定策略的副本
        self.actor_fixed = self.actor.to(self.device)
        for param in self.actor_fixed.parameters():
            param.requires_grad = False
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.n_updates += 1
        self.beta *= math.exp(-self.lambda_decay * self.n_updates)
        
        for _ in range(gradient_steps):
            high_batch, low_batch = self.replay_buffer.sample_split(batch_size)
            
            if not high_batch or not low_batch:
                continue
            
            self._update_critic(high_batch)
            self._update_critic_low(low_batch)
            self._update_discriminator(high_batch)
            self._update_actor_with_mi(high_batch, low_batch)
            self._update_target_networks()
        
        # 每1000次更新后更新固定策略
        if self.n_updates % 1000 == 0:
            self._update_fixed_policy()
    
    def _update_fixed_policy(self):
        """定期更新固定策略"""
        for target_param, param in zip(self.actor_fixed.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
    
    def _update_critic_low(self, batch: Dict[str, th.Tensor]) -> None:
        with th.no_grad():
            next_actions = self.actor(batch["next_observations"])
            next_q_values = self.critic_low_target(batch["next_observations"], next_actions)
            next_q_values = th.max(next_q_values, dim=1, keepdim=True)[0]
            target_q_values = batch["rewards"] + (1 - batch["dones"]) * self.gamma * next_q_values
        
        current_q_values = self.critic_low(batch["observations"], batch["actions"])
        critic_loss = F.mse_loss(current_q_values, target_q_values.repeat(1, current_q_values.shape[1]))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def _update_discriminator(self, batch: Dict[str, th.Tensor]) -> None:
        obs = batch["observations"]
        
        with th.no_grad():
            pos_actions = self.actor(obs)
        pos_logits = self.discriminator(obs, pos_actions)
        
        # 使用更多的负样本来提高判别器的鲁棒性
        neg_batch_size = min(pos_actions.shape[0] * 2, 512)
        neg_actions = self.replay_buffer.sample_negative_actions(obs, batch_size=neg_batch_size)
        neg_logits = self.discriminator(obs, neg_actions)
        
        # 使用标签平滑技术提高稳定性
        pos_target = th.ones_like(pos_logits) * 0.9  # 标签平滑
        neg_target = th.zeros_like(neg_logits) + 0.1  # 标签平滑
        
        pos_loss = F.binary_cross_entropy(pos_logits, pos_target)
        neg_loss = F.binary_cross_entropy(neg_logits, neg_target)
        discriminator_loss = pos_loss + neg_loss
        
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        # 梯度裁剪以提高稳定性
        th.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.discriminator_optimizer.step()
        
        # 记录判别器性能
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record("train/discriminator_loss", discriminator_loss.item())
            self.logger.record("train/discriminator_accuracy", 
                            (th.mean((pos_logits > 0.5).float()) + 
                             th.mean((neg_logits < 0.5).float())) / 2)
    
    def _update_actor_with_mi(self, high_batch: Dict[str, th.Tensor], low_batch: Dict[str, th.Tensor]) -> None:
        high_obs = high_batch["observations"]
        low_obs = low_batch["observations"]
        
        actions = self.actor(high_obs)
        q_high = self.critic.q1_forward(high_obs, actions)
        
        with th.no_grad():
            fixed_actions = self.actor_fixed(low_obs)
            current_actions = self.actor(low_obs)
            
            q_low_fixed = self.critic.q1_forward(low_obs, fixed_actions)
            q_low_current = self.critic_low.q1_forward(low_obs, current_actions)
            q_low = th.max(q_low_fixed, q_low_current)
            
            # 使用更稳定的行为差距计算
            behavior_gap = th.clamp(q_high.mean() - q_low.mean(), -10.0, 10.0)
        
        # 使用更稳定的互信息计算
        mi_logits = self.discriminator(high_obs, actions)
        mi_term = th.log(mi_logits + 1e-8).mean()
        
        log_prob = self.actor.log_prob(actions)
        if isinstance(self.ent_coef, th.Tensor):
            entropy_loss = -th.mean(self.ent_coef * log_prob)
        else:
            entropy_loss = -th.mean(self.ent_coef * log_prob)
        
        # 动态调整各项权重
        actor_loss = -(q_high.mean() + behavior_gap + self.beta * mi_term + entropy_loss)
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        # 添加梯度裁剪
        th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor.optimizer.step()
        
        # 记录各项损失
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record("train/actor_q_value", q_high.mean().item())
            self.logger.record("train/behavior_gap", behavior_gap.item())
            self.logger.record("train/mi_term", mi_term.item())
            self.logger.record("train/beta", self.beta)
    
    def _update_target_networks(self) -> None:
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic_low.parameters(), self.critic_low_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def collect_rollouts(self, *args, **kwargs):
        result = super().collect_rollouts(*args, **kwargs)
        return result
