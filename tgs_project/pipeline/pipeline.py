from __future__ import annotations
from typing import Callable, Iterable, Any, List, Union
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm


class Stage:
    """Wrap a callable so it can be chained with the `|` operator."""
    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    # data | Stage
    def __ror__(self, other: Any) -> Any:
        if isinstance(other, Iterable):
            # If the other is an iterable, we need to apply the function to each item
            return [self.fn(item) for item in other]
        return self.fn(other)

    # StageA | StageB   ->   Pipeline(StageA, StageB)
    def __or__(self, other: Union[Stage, "Pipeline"]) -> "Pipeline":
        return Pipeline([self]) | other


class Pipeline(Stage):
    def __init__(self, stages: List[Stage] | None = None):
        self.stages = stages or []
        super().__init__(self)  # Now pickleable

    def __call__(self, x):
        return self.__ror__(x)

    def __or__(self, other: Union[Stage, "Pipeline"]) -> "Pipeline":
        return Pipeline(self.stages + (other.stages if isinstance(other, Pipeline) else [other]))

    def __ror__(self, other: Any) -> Any:
        """Apply the pipeline to single data or list of data."""
        if isinstance(other, Iterable) and not isinstance(other, (str, bytes, dict)):
            return [self.__ror__(item) for item in other]  # recursively apply to each
        data = other
        for stage in self.stages:
            data = stage.fn(data)
        return data
    
class Parallel(Stage):
    def __init__(
        self,
        stage: Stage,
        n_workers: int | None = None,
        chunksize: int = 1,
        initializer: Callable[[], None] | None = None,
    ):
        self.stage = stage
        self.n_workers = n_workers or mp.cpu_count()
        self.chunksize = chunksize
        self.initializer = initializer

    def __ror__(self, data: Iterable[Any]) -> list[Any]:
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=self.initializer
        ) as pool:
            return list(tqdm(
                pool.map(self.stage, data, chunksize=self.chunksize), # type: ignore
                total=len(data),  # type: ignore
                desc="Processing"
            ))