# play_checkpoint.py
#학습이 체크포인트를 저장한 뒤, 그 체크포인트를 불러서 seeker/runner를 실제로 움직이게 해서 
#ASCII로 맵을 프레임처럼 출력해줄 수 있어(나중에 pygame로도 가능).

import os
import time
import argparse
import hide_seek_env_los
import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from hide_seek_env_los import HideSeekLOS

MAP = r"""
##############################################################
#                  #         #      #          #             #
#                  #         #      #          #             #
#  S               #         #      #          #             #
########           #                #          #             #
# H                ###########      #    #######             #
#                                   #          #             #
#      ###############              #          ###########   #
#         R   #            ##########          b             #
#             #                                              #
#             #                                              #
#             #       ################################       #
#             #              #                               #
#             #              #               #  ##############
#             #              #               #               #
#             b              #               #               #
#                            #               #               #
#                                            #               #
#                                            #               #
####.########d##########.########d############################
#         #         #         #          d                   #
#   b     #         #         #          #                   #
#         #         #         #          #                   #
#         #         #         #          #                   #
##############################################################
"""

def env_creator(env_config):
    env = HideSeekLOS(
        map_str=env_config["map_str"],
        stage=env_config["stage"],
        door_hp=env_config.get("door_hp", 6),
        sight=env_config.get("sight", 8),
        max_steps=env_config.get("max_steps", 500),
        step_penalty=env_config.get("step_penalty", -0.001),
        catch_dist=env_config.get("catch_dist", 1),
        los_blocks_by_boxes=env_config.get("los_blocks_by_boxes", True),
        resolve_order=env_config.get("resolve_order", "random"),
    )
    return ParallelPettingZooEnv(env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    register_env("hide_seek_los", env_creator)

    # 체크포인트 로드
    algo = PPO.from_checkpoint(args.checkpoint)

    # 원본 환경(ASCII 렌더 쓰려고 래핑 전 env도 같이 생성)
    raw_env = HideSeekLOS(map_str=MAP, stage=args.stage, sight=8, max_steps=500, resolve_order="random")
    obs, _ = raw_env.reset()

    dt = 1.0 / max(args.fps, 1e-6)

    for t in range(args.steps):
        # 각 에이전트 관측으로 각 정책 action 계산
        seeker_obs = obs["seeker"]
        runner_obs = obs["runner"]

        seeker_action = algo.compute_single_action(seeker_obs, policy_id="seeker_policy")
        runner_action = algo.compute_single_action(runner_obs, policy_id="runner_policy")

        obs, rewards, terminated, truncated, infos = raw_env.step({
            "seeker": int(seeker_action),
            "runner": int(runner_action),
        })

        os.system("cls" if os.name == "nt" else "clear")
        print(f"t={t} stage={args.stage} rewards={rewards} term={terminated} trunc={truncated}")
        print(raw_env.render_ascii())

        if any(terminated.values()) or any(truncated.values()):
            print("Episode finished.")
            break

        time.sleep(dt)

    ray.shutdown()
    import hide_seek_env_los
    print("ENV FILE =", hide_seek_env_los.__file__)
if __name__ == "__main__":
    main()
