from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


"""
    获取人脸交换器
"""
def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # 模型路径
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            # 加载模型， INSwapper
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    # 检查模型是否已经下载
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


"""
    换脸
"""
def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    # 获取 换脸器
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


"""
    处理图像帧
"""
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    # 是否开启多人脸替换
    if roop.globals.many_faces:
        # 解析图像中的所有人脸
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            # 循环替换人脸
            for target_face in many_faces:
                # 换脸
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        # 单人脸替换，解析图像中相似的人脸
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            # 换脸
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


"""
    批量处理图像帧
"""
def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


"""
    处理图像
"""
def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # 从源文件中解析一张人脸
    source_face = get_one_face(cv2.imread(source_path))
    # 读取目标图像
    target_frame = cv2.imread(target_path)
    # 从目标图像中解析参考人脸
    reference_face = get_one_face(target_frame, roop.globals.reference_face_position)
    # 替换 人脸
    result = process_frame(source_face, reference_face, target_frame)
    # 显示处理完成的图像
    cv2.imwrite(output_path, result)


"""
    处理视频
"""
def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not get_face_reference():
        # 读取参考帧
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        # 从参考帧中的 指定位置 解析出一个人脸
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        # 设置参考帧
        set_face_reference(reference_face)
    # 处理视频数据
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
