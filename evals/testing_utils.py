import sys
import tempfile
from pathlib import Path
import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession

def build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))

def make_session(memory_root: Path | None = None, env_id: str = "MiniGrid-GoToDoor-8x8-v0", seed: int = 42, **kwargs) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id=env_id,
        seed=seed,
        render_mode="none",
        memory_root=memory_root or Path(tempfile.mkdtemp()),
        **kwargs
    )
