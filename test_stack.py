import robosuite as suite
from robosuite.wrappers import GymWrapper
import gymnasium as gym
from stable_baselines3.ppo import PPO

if __name__ == "__main__":

    env = GymWrapper(
        suite.make(
            "Stack",
            robots="Panda",    # 机器人类型
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=True,
            reward_shaping=True,
            control_freq=20,
        )
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("stack_ppo")

    # model.load("stack_ppo")

    # # 只取obs，丢弃info
    # obs, _ = env.reset()

    # while True:
    #     action, _states = model.predict(obs)  
    #     # 在Gymnasium里，step通常返回五个量
    #     obs, reward, done, truncated, info = env.step(action)
    #     env.render()

    #     # 如果done或truncated，需要reset
    #     if done or truncated:
    #         obs, _ = env.reset()