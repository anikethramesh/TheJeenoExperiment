from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Observation:
    raw: dict[str, Any]
    step_count: int


class MiniGridAdapter:
    def __init__(self, env):
        self.env = env
        self.obs: dict[str, Any] | None = None
        self.info: dict[str, Any] | None = None
        self.step_count = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.obs, self.info = self.env.reset(seed=seed)
        self.step_count = 0
        return Observation(raw=self.obs, step_count=self.step_count)

    def observe(self) -> Observation:
        if self.obs is None:
            raise RuntimeError("Environment has not been reset yet.")
        return Observation(raw=self.obs, step_count=self.step_count)

    def list_grid_objects(self) -> list[dict[str, Any]]:
        raw_env = self.env.unwrapped
        objects: list[dict[str, Any]] = []
        for x in range(raw_env.width):
            for y in range(raw_env.height):
                cell = raw_env.grid.get(x, y)
                if cell is None:
                    continue
                cell_type = getattr(cell, "type", None)
                if cell_type is None:
                    continue
                objects.append(
                    {
                        "type": cell_type,
                        "color": getattr(cell, "color", None),
                        "x": x,
                        "y": y,
                    }
                )
        return objects

    def find_matching_objects(
        self,
        object_type: str | None = None,
        color: str | None = None,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for obj in self.list_grid_objects():
            if object_type is not None and obj["type"] != object_type:
                continue
            if color is not None and obj["color"] != color:
                continue
            matches.append(obj)
        return matches

    def retarget_to_object(self, target_object: dict[str, Any]) -> dict[str, Any]:
        raw_env = self.env.unwrapped
        target_pos = (int(target_object["x"]), int(target_object["y"]))

        if hasattr(raw_env, "target_pos"):
            raw_env.target_pos = target_pos
        if hasattr(raw_env, "target_color") and target_object.get("color") is not None:
            raw_env.target_color = target_object["color"]

        color = target_object.get("color")
        object_type = target_object.get("type")
        if color and object_type:
            raw_env.mission = f"go to the {color} {object_type}"
        elif object_type:
            raw_env.mission = f"go to the {object_type}"

        if self.obs is not None:
            self.obs["mission"] = getattr(raw_env, "mission", self.obs.get("mission"))

        return {
            "type": object_type,
            "color": color,
            "x": target_pos[0],
            "y": target_pos[1],
        }

    def act(self, action: int):
        if self.obs is None:
            raise RuntimeError("Environment has not been reset yet.")
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.info = info
        observation = Observation(raw=self.obs, step_count=self.step_count)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
