from __future__ import annotations
from typing import Callable, Iterable, Any, List, Union
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass, field
from collections import deque

from itertools import islice
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

def batched(it: Iterable[T], size: int) -> Iterator[list[T]]:
    """
    Yield successive batches of size `size` from `it`.
    Works with any iterable (lists, generators, etc.).
    """
    it = iter(it)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

@dataclass
class DataBatch:
    """
    Wrapper for streaming or batched data with optional count and deduplication support.
    """
    data: Iterable[Any]
    count: int|None = None
    processed_ids: set[str] = field(default_factory=set)

    def __iter__(self):
        # Filter on-the-fly when iterating
        return (item for item in self.data if item.get("id") not in self.processed_ids)
    def batched(self, size: int) -> Iterator[list[Any]]:
        return batched(self, size)
    
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
            return (self.__ror__(item) for item in other)  # recursively apply to each
        data = other
        for stage in self.stages:
            data = stage.fn(data)
        return data
    
class Parallel(Stage):
    def __init__(
        self,
        stage: Stage,
        n_workers: int = -1,
        chunksize: int = 1,
        initializer: Callable[[], None] | None = None,
    ):
        self.stage = stage
        self.n_workers = n_workers if n_workers > 0 else mp.cpu_count()
        self.chunksize = chunksize
        self.initializer = initializer
        super().__init__(self._stream)  # delegate __call__ to streaming logic

    def _stream(self, data: Iterable[Any]) -> Iterable[Any]:
        if not isinstance(data, DataBatch):
            try:
                count = len(data)  # type: ignore
            except TypeError:
                count = None
            data = DataBatch(data, count=count)

        max_in_flight = self.n_workers * self.chunksize
        iterator = iter(data)
        in_flight = deque()

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=self.initializer
        ) as pool:
            pbar = tqdm(total=data.count or None, desc="Processing")
            try:
                # Prime the queue
                for _ in range(max_in_flight):
                    try:
                        item = next(iterator)
                        fut = pool.submit(self.stage.fn, item)
                        in_flight.append(fut)
                    except StopIteration:
                        break

                # Yield results and refill
                while in_flight:
                    fut = in_flight.popleft()
                    yield fut.result()
                    pbar.update(1)

                    try:
                        item = next(iterator)
                        fut = pool.submit(self.stage.fn, item)
                        in_flight.append(fut)
                    except StopIteration:
                        pass
            finally:
                pbar.close()

    def __ror__(self, data: Iterable[Any]) -> Iterable[Any]:
        return self._stream(data)