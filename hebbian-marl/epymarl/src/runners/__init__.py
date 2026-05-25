REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .hebbian_runner import HebbianRunner
REGISTRY["hebbian"] = HebbianRunner

from .hebbian_parallel_runner import HebbianParallelRunner
REGISTRY["hebbian_parallel"] = HebbianParallelRunner
