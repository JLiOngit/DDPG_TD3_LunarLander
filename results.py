"""Saving the results of each run (`results.json` in its directory) and loading them back

A `results.json` file holds the name of the run (e.g. `td3/layer_norm=True/S1`), its full
configuration, the rewards of each evaluation and the bias estimates. Loading the files
gives back the `(history, bias_history)` pairs returned by `run_one`, so that the analysis
works the same on runs that just finished and on runs saved earlier.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from rl_mind.evaluation import EvalResult

from config import DDPGConfig


def save_results(run_dir: Path, name: str, cfg: DDPGConfig, history: list[EvalResult],
                 bias_history: list[tuple[int, dict[str, float]]], done: bool = False) -> None:
    """Write (or overwrite) `results.json` in `run_dir`

    :param done: True once the run is complete (`load_runs` ignores the other runs)
    """
    data = {
        "name": name,
        "config": asdict(cfg),
        "done": done,
        "evals": [{"step": result.step, "rewards": result.rewards.tolist()} for result in history],
        "bias": [{"step": step, **bias} for step, bias in bias_history],
    }
    path = Path(run_dir) / "results.json"
    # Write then rename, so that an interrupted run never leaves a half-written file
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def load_runs(root: str = "outputs", only_done: bool = True) -> dict[str, list]:
    """The saved runs, grouped by variant (`"ddpg/base"`, `"td3/layer_norm=True"`, ...)

    Each run is a `(history, bias_history)` pair, as returned by `run_one`. When the same
    run (variant and seed) was launched several times, the most recent one is kept.
    """
    latest = {}
    # Directory names end with the date: the most recent run of a name comes last
    for path in sorted(Path(root).glob("**/results.json")):
        data = json.loads(path.read_text())
        if data["done"] or not only_done:
            latest[data["name"]] = data

    runs: dict[str, list] = {}
    for name, data in sorted(latest.items()):
        history = [EvalResult(e["step"], torch.tensor(e["rewards"]), False) for e in data["evals"]]
        bias_history = [(bias.pop("step"), bias) for bias in data["bias"]]
        runs.setdefault(name.rsplit("/", 1)[0], []).append((history, bias_history))
    return runs


def results_from_tensorboard(run_dir: Path, name: str, cfg: DDPGConfig) -> None:
    """Write `results.json` for a run that was logged in tensorboard only

    Tensorboard only holds the mean reward of each evaluation: it becomes the single
    value of `rewards`.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    events.Reload()
    history = [EvalResult(e.step, torch.tensor([e.value]), False) for e in events.Scalars("eval/reward")]
    bias_by_step: dict[int, dict[str, float]] = {}
    for tag in events.Tags()["scalars"]:
        if tag.startswith("bias/"):
            for e in events.Scalars(tag):
                bias_by_step.setdefault(e.step, {})[tag.removeprefix("bias/")] = e.value
    done = history[-1].step >= cfg.max_steps
    save_results(run_dir, name, cfg, history, sorted(bias_by_step.items()), done=done)
