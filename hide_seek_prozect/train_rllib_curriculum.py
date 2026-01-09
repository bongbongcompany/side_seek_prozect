# train_rllib_curriculum.py
"""
- RLlib PPO self-play (seeker/runner 동시 학습)
- LOS + shaping(위험반경/LOS/탐색보상) 사용
- 커리큘럼:
  stage 0 -> 1 -> 2 -> 3
  + stage2에서는 “입구 1칸 방” 난이도를 줄이려고 맵을 살짝 넓힌 버전 사용
  + stage3에서 원본 맵으로 복귀
- 개선:
  1) 포획/생존(시간초과) 보상 파라미터화 + 포획 보상 상향
  2) RLlib Custom Callback으로 목표 보상 도달 시 다음 스테이지로 조기 진행
"""

from __future__ import annotations

import os
import time
import glob
from typing import Dict, Optional
import argparse

import ray
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env

from hide_seek_env_los import HideSeekLOS


# ===== 원본 맵 =====
MAP_ORIG = r"""
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

# ===== stage2 완화 맵 =====
MAP_STAGE2_EASIER = r"""
##############################################################
#                  #                           #             #
#                  #                           #             #
#  S               #                           #             #
#                  #                                         #
#                                              #             #
#                                              #             #
#      ###############                         ###########   #
#         R                                                  #
##############################################################
"""
# NOTE: '.'는 “빈칸 취급”이므로 통로 확장용으로 사용 가능(네 env는 '.'를 빈칸 처리)


def env_creator(env_config: Dict):
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
        # shaping
        danger_r=env_config.get("danger_r", 6),
        danger_coef=env_config.get("danger_coef", 0.01),
        see_reward=env_config.get("see_reward", 0.02),
        see_penalty=env_config.get("see_penalty", 0.02),
        explore_reward=env_config.get("explore_reward", 0.003),
        # terminal rewards (NEW)
        catch_reward=env_config.get("catch_reward", 5.0),
        caught_penalty=env_config.get("caught_penalty", 5.0),
        escape_reward=env_config.get("escape_reward", 1.0),
        escape_penalty=env_config.get("escape_penalty", 1.0),
        # seeker freeze
        seeker_freeze_steps=env_config.get("seeker_freeze_steps", 0),
    )
    return ParallelPettingZooEnv(env)


class CurriculumCallback(DefaultCallbacks):
    """최근 N회 평균 보상이 목표를 넘으면 stage를 조기 종료시키기 위한 플래그를 result에 추가."""

    def __init__(self):
        super().__init__()
        self.hist = []
        self.window = None
        self.passed = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        target = algorithm.config.get("curriculum_target_reward", None)
        window = int(algorithm.config.get("curriculum_window", 10))
        patience = int(algorithm.config.get("curriculum_patience", 3))

        if target is None:
            return

        # stage별로 window가 바뀌면 히스토리 리셋
        if self.window != window:
            self.window = window
            self.hist = []
            self.passed = 0

        rew = result.get("episode_reward_mean", None)
        if rew is None:
            return

        self.hist.append(float(rew))
        if len(self.hist) > window:
            self.hist = self.hist[-window:]

        if len(self.hist) < window:
            result.setdefault("custom_metrics", {})
            result["custom_metrics"]["curriculum_avg_rew"] = float(sum(self.hist) / max(1, len(self.hist)))
            result["custom_metrics"]["curriculum_advance"] = 0.0
            result["curriculum_advance"] = False
            return

        avg = float(sum(self.hist) / window)

        if avg >= float(target):
            self.passed += 1
        else:
            self.passed = 0

        advance = (self.passed >= patience)

        result.setdefault("custom_metrics", {})
        result["custom_metrics"]["curriculum_avg_rew"] = avg
        result["custom_metrics"]["curriculum_advance"] = 1.0 if advance else 0.0
        result["curriculum_advance"] = advance


def _is_rllib_checkpoint_dir(p: str) -> bool:
    """Ray/RLlib 체크포인트 디렉토리인지(대략) 판별."""
    if not os.path.isdir(p):
        return False
    # Ray 버전별로 파일명이 조금씩 달라서 넓게 체크
    candidates = [
        os.path.join(p, "algorithm_state.pkl"),
        os.path.join(p, "rllib_checkpoint.json"),
        os.path.join(p, "checkpoint.json"),
    ]
    if any(os.path.exists(x) for x in candidates):
        return True
    # 혹시 파일명이 다르더라도 무언가 checkpoint-* 형태가 있으면 인정
    if glob.glob(os.path.join(p, "checkpoint-*")):
        return True
    return False


def find_latest_checkpoint(path: str) -> Optional[str]:
    """
    path가
    - checkpoint_000xxx 디렉토리면 그대로 반환
    - 상위 폴더(checkpoints_hide_seek)이면 그 안의 checkpoint_* 중 최신 반환
    - 아무것도 못 찾으면 None
    """
    path = os.path.abspath(path)

    # 1) 사용자가 checkpoint_000xxx 같은 디렉토리를 직접 준 경우
    if _is_rllib_checkpoint_dir(path):
        return path

    # 2) 폴더를 준 경우 내부에서 checkpoint_* 찾기
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "checkpoint_*"))
        cands = [c for c in cands if _is_rllib_checkpoint_dir(c)]
        if not cands:
            return None
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return cands[0]

    return None

def find_latest_checkpoint(path: str) -> Optional[str]:
    """
    path가
    - checkpoint_000xxx 디렉토리면 그대로 반환
    - 상위 폴더(checkpoints_hide_seek)이면 그 안의 checkpoint_* 중 최신 반환
    - 아무것도 못 찾으면 None
    """
    path = os.path.abspath(path)

    # 1) 사용자가 checkpoint_000xxx 같은 디렉토리를 직접 준 경우
    if _is_rllib_checkpoint_dir(path):
        return path

    # 2) 폴더를 준 경우 내부에서 checkpoint_* 찾기
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "checkpoint_*"))
        cands = [c for c in cands if _is_rllib_checkpoint_dir(c)]
        if not cands:
            return None
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return cands[0]

    return None


def main(checkpoint_to_restore: Optional[str] = None):
    import os
    import time
    from typing import Dict, Optional

    import ray
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    ray.init(ignore_reinit_error=True)

    # env 등록
    register_env("hide_seek_los", env_creator)

    # multi-agent policies
    policies = {
        "seeker_policy": PolicySpec(),
        "runner_policy": PolicySpec(),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "seeker_policy" if agent_id == "seeker" else "runner_policy"

    # 커리큘럼
    curriculum = [
        {"stage": 0, "max_iters": 40,  "min_iters": 10, "target_reward": 0.20},
        {"stage": 1, "max_iters": 60,  "min_iters": 15, "target_reward": 0.30},
        {"stage": 2, "max_iters": 80,  "min_iters": 20, "target_reward": 0.40},
        {"stage": 3, "max_iters": 120, "min_iters": 30, "target_reward": 0.50},
    ]

    #  CNN 전용 체크포인트 폴더 (기존 FC 체크포인트와 섞이지 않게!)
    out_dir = os.path.abspath("./checkpoints_hide_seek_cnn")
    os.makedirs(out_dir, exist_ok=True)

    #  자동 restore 처리 (CNN 폴더 기준)
    # - 사용자가 --restore를 주면 그 경로에서 최신 ckpt를 찾고
    # - 안 주면 out_dir에서 최신 ckpt를 자동 탐색
    if checkpoint_to_restore:
        auto_ckpt = find_latest_checkpoint(checkpoint_to_restore)
        if auto_ckpt is None:
            print(f"[WARN] --restore 경로에서 체크포인트를 찾지 못함: {checkpoint_to_restore}")
            checkpoint_to_restore = None
        else:
            checkpoint_to_restore = auto_ckpt
            print(f"[INFO] Auto-restore from user path: {checkpoint_to_restore}")
    else:
        auto_ckpt = find_latest_checkpoint(out_dir)
        if auto_ckpt:
            checkpoint_to_restore = auto_ckpt
            print(f"[INFO] Auto-restore latest checkpoint in out_dir: {checkpoint_to_restore}")
        else:
            print("[INFO] No CNN checkpoint found. Training from scratch.")

    # ===== stage loop =====
    for item in curriculum:
        stage = item["stage"]
        max_iters = item["max_iters"]
        min_iters = item.get("min_iters", 0)
        target_reward = item.get("target_reward", None)

        # stage2에서는 완화 맵 사용, stage3에서 원본으로 복귀
        map_for_stage = MAP_STAGE2_EASIER if stage == 2 else MAP_ORIG

        print(f"\n========== CURRICULUM STAGE {stage} (max_iters={max_iters}, target={target_reward}) ==========")

        env_cfg: Dict = {
            "map_str": map_for_stage,
            "stage": stage,
            "door_hp": 6,
            "sight": 8,
            "max_steps": 2500,
            "step_penalty": -0.001,
            "catch_dist": 1,
            "los_blocks_by_boxes": True,
            "resolve_order": "random",

            # shaping
            "danger_r": 6,
            "danger_coef": 0.01,
            "see_reward": 0.02,
            "see_penalty": 0.02,
            "explore_reward": 0.003,

            # terminal rewards
            "catch_reward": 8.0 if stage >= 3 else 5.0,
            "caught_penalty": 8.0 if stage >= 3 else 5.0,
            "escape_reward": 1.0,
            "escape_penalty": 1.0,

            # seeker freeze
            "seeker_freeze_steps": 300 if stage <= 1 else 0,
        }

        config = (
            PPOConfig()
            .environment(env="hide_seek_los", env_config=env_cfg)
            .framework("torch")
            .rollouts(num_rollout_workers=2)
            .training(
                gamma=0.99,
                lr=3e-4,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.0,

                #  PPO 안정화(권장)
                train_batch_size=4000,
                sgd_minibatch_size=256,
                num_sgd_iter=10,

                #  CNN 모델(HWC: 17x17x5)용 설정
                model={
                    "conv_filters": [
                        [32, [3, 3], 1],
                        [64, [3, 3], 1],
                        [64, [3, 3], 1],
                    ],
                    "conv_activation": "relu",
                    "post_fcnet_hiddens": [256, 256],
                    "post_fcnet_activation": "relu",

                    # vf_share_layers는 RL 모델의 두 핵심 축인 **Policy(행동 결정)**와 **Value Function(가치 평가)**이 
                    # 관측 데이터로부터 특징을 추출하는 '두뇌의 앞부분(Backbone)'을 함께 쓸 것인지를 결정하는 옵션입니다.
                    #결론부터 말씀드리면, 지금처럼 CNN을 사용하여 맵의 시각적 정보를 파악해야 하는 경우 True로 두는 것이 훨씬 효율적입니다.
                    "vf_share_layers": True,
                },
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["seeker_policy", "runner_policy"],
            )
            .callbacks(CurriculumCallback)
            .experimental(_disable_preprocessor_api=True)
        )

        # callback 조건
        config = config.update_from_dict(
            {
                "curriculum_target_reward": target_reward,
                "curriculum_window": 10,
                "curriculum_patience": 3,
            }
        )

        algo = config.build()

        #  자동 restore: stage0부터도 이어서 시작(동일 CNN 구조일 때만)
        if checkpoint_to_restore:
            print("[INFO] Restoring from:", checkpoint_to_restore)
            algo.restore(checkpoint_to_restore)

        # 학습(조기 종료 포함)
        for i in range(max_iters):
            result = algo.train()

            it = result["training_iteration"]
            rew_mean = result.get("episode_reward_mean", None)
            len_mean = result.get("episode_len_mean", None)

            if (it % 5 == 0) and (rew_mean is not None) and (len_mean is not None):
                avg = result.get("custom_metrics", {}).get("curriculum_avg_rew", None)
                print(
                    f"[stage {stage} | it {it:4d}] "
                    f"ep_rew_mean={rew_mean:.4f} ep_len_mean={len_mean:.1f} "
                    f"curr_avg={avg}"
                )

            if (i + 1) >= min_iters and result.get("curriculum_advance", False):
                avg = result.get("custom_metrics", {}).get("curriculum_avg_rew", None)
                print(f" Stage {stage} target reached (avg={avg}). Advancing early at iter={it}")
                break

        # 체크포인트 저장 (CNN 폴더에 저장)
        ckpt = algo.save(out_dir)
        print("[INFO] Saved checkpoint:", ckpt)

        # 다음 stage는 방금 저장한 ckpt로 이어서
        checkpoint_to_restore = ckpt

        algo.stop()
        time.sleep(1)

    print("\nDone. Final checkpoint:", checkpoint_to_restore)
    ray.shutdown()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=str, default=None, help="checkpoint 경로(파일)로부터 복구")
    args = parser.parse_args()
    main(checkpoint_to_restore=args.restore)
