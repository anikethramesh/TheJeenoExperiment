from __future__ import annotations

from gymnasium.envs.registration import register, registry


CUSTOM_GOTODOOR_SIZES = (10, 12, 16)


def ensure_custom_minigrid_envs_registered() -> None:
    for size in CUSTOM_GOTODOOR_SIZES:
        env_id = f"MiniGrid-GoToDoor-{size}x{size}-v0"
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="minigrid.envs:GoToDoorEnv",
            kwargs={"size": size},
        )
