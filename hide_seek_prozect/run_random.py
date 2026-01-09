"""
run_random.py

환경이 제대로 동작하는지 확인용:
- 랜덤 행동으로 몇 스텝 굴려보고
- ASCII 렌더로 상태를 출력한다.
"""

import time
import numpy as np

from hide_seek_env import HideSeekEnv


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

def main():
    env = HideSeekEnv(
    map_str=MAP,
    door_hp=5,
    sight=6,
    max_steps=200,
    step_penalty=-0.001,   # <- 술래는 빨리잡기 / 숨는자는 오래버티기
    resolve_order="random",
    )

    obs, infos = env.reset()
    print(env.render_ascii())
    print("-" * 60)

    rng = np.random.default_rng(0)

    for t in range(200):
        actions = {
            "seeker": int(rng.integers(0, 7)),
            "runner": int(rng.integers(0, 7)),
        }
        obs, rewards, terminated, truncated, infos = env.step(actions)

        # 화면 출력(너무 빠르면 보기 힘드니 약간 쉬기)
        print(f"[t={t}] actions={actions} rewards={rewards} term={terminated} trunc={truncated}")
        print(env.render_ascii())
        print("-" * 60)
        time.sleep(0.05)

        if any(terminated.values()) or any(truncated.values()):
            print("Episode finished.")
            break


if __name__ == "__main__":
    main()