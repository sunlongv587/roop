import threading
from typing import Any, Optional, List
import insightface
import numpy

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

"""
    获取人脸解析器
"""
def get_face_analyser() -> Any:
    global FACE_ANALYSER
    # 线程加锁
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # https://insightface.ai/
            # 构建人脸分析器，param2 = 是执行器（GPU or CPU）
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            # 猜测可能是 CUDD:0 执行器 ID
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


"""
    删除人脸解析器
"""
def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None


"""
    解析所有人脸，只读取一个，默认取第一个
"""
def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


"""
    解析该图像中的所有人脸
"""
def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        # 使用 人脸分析器，解析图像帧中的人脸信息
        return get_face_analyser().get(frame)
    except ValueError:
        return None


"""
    从图像中查找 相似的人脸
"""
def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    # 从图像中解析所有人脸
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            # 解析人脸中的 normed_embedding， 做对比
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                # 相减，平方，再相加，计算 距离
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                # 如果距离小于 配置文件中的阈值，就 表示相似
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
