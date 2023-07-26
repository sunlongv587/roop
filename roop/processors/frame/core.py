import os
import sys
import importlib
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import roop

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_frames',
    'process_image',
    'process_video',
    'post_process'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'roop.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError:
        sys.exit(f'Frame processor {frame_processor} not found.')
    except NotImplementedError:
        sys.exit(f'Frame processor {frame_processor} not implemented correctly.')
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES


"""
    多线程处理
"""
def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None]) -> None:
    # 线程池
    with ThreadPoolExecutor(max_workers=roop.globals.execution_threads) as executor:
        # 执行结果集
        futures = []
        # 使用数据列表 构建 队列
        queue = create_queue(temp_frame_paths)
        # 每个线程分多少帧数据，计算每个线程处理的数据 个数，按照帧数 / 线程数
        queue_per_future = max(len(temp_frame_paths) // roop.globals.execution_threads, 1)
        while not queue.empty():
            # 提交到线程池处理
            future = executor.submit(process_frames, source_path, pick_queue(queue, queue_per_future), update)
            futures.append(future)
        for future in as_completed(futures):
            future.result()


"""
    创建队列
"""
def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


"""
    从队列中取出若干数据
"""
def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


"""
    处理视频数据
"""
def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    # 进度条？
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        # 多线程处理帧数据
        multi_process_frame(source_path, frame_paths, process_frames, lambda: update_progress(progress))


"""
    更新进度条
"""
def update_progress(progress: Any = None) -> None:
    process = psutil.Process(os.getpid())
    # 内存使用情况
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    progress.set_postfix({
        'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
        # 空闲线程数
        'execution_providers': roop.globals.execution_providers,
        # 使用线程数
        'execution_threads': roop.globals.execution_threads
    })
    progress.refresh()
    progress.update(1)
