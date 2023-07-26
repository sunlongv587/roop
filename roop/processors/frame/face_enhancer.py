from typing import Any, List, Callable
import cv2
import threading
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

# 图像增强器
def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            # 读取 并加载 GFPGAN模型
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            # 构建 图像增强器，（盲人面部恢复器？）
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER


# 设备检测
def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER

    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


# 增强脸部
def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    # 获取人脸框选坐标
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    # 将临时帧中的 人脸抠出来
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        # 线程映射
        with THREAD_SEMAPHORE:
            # 增强人脸，贴回去
            _, _, temp_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )
        # 貌似这就是把脸换掉了
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame


"""
    处理图像帧
"""
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


"""
    处理视频的每一帧
"""
def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    # 遍历视频帧
    for temp_frame_path in temp_frame_paths:
        # 使用openCV 加载 图像帧
        temp_frame = cv2.imread(temp_frame_path)
        # 处理图像帧
        result = process_frame(None, None, temp_frame)
        # 将处理后的帧显示出来
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


"""
    处理图像换脸
"""
def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # openCV 读取图像
    target_frame = cv2.imread(target_path)
    # 处理图像帧
    result = process_frame(None, None, target_frame)
    # 显示图像帧
    cv2.imwrite(output_path, result)


"""
    处理视频
"""
def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
