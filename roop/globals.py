from typing import List

source_path = None
target_path = None
output_path = None
headless = None
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
skip_audio = None
# 多人脸替换开关
many_faces = None
reference_face_position = None
# 引用/参考 帧数
reference_frame_number = None
similar_face_distance = None
temp_frame_format = None
temp_frame_quality = None
output_video_encoder = None
output_video_quality = None
max_memory = None
execution_providers: List[str] = []
# 最大线程数
execution_threads = None
log_level = 'error'
