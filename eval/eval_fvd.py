# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import os
import numpy as np
import torch
import tqdm
from eval.fvd import compute_our_fvd
from latentsync.utils.face_detector import FaceDetector
from latentsync.utils.media import read_video_frames


class FVD:
    def __init__(self, resolution=(224, 224)):
        self.face_detector = FaceDetector("cpu")
        self.resolution = resolution

    def detect_face(self, image):
        bbox, _ = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")
        xmin, ymin, xmax, ymax = bbox
        image = image[ymin:ymax, xmin:xmax]

        return image

    def detect_video(self, video_path):
        video_frames = read_video_frames(video_path, range(20, 36))
        faces = []
        for frame in video_frames:
            face = self.detect_face(frame)
            face = cv2.resize(face, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            faces.append(face)

        if len(faces) != 16:
            return RuntimeError("Insufficient consecutive frames of faces (less than 16).")
        faces = np.stack(faces, axis=0)  # (f, h, w, c)
        faces = torch.from_numpy(faces)
        return faces

    def detect_videos(self, videos_dir: str):
        videos_list = []

        if videos_dir.endswith(".mp4"):
            video_faces = self.detect_video(videos_dir)
            videos_list.append(video_faces)
        else:
            for file in tqdm.tqdm(os.listdir(videos_dir)):
                if file.endswith(".mp4"):
                    video_path = os.path.join(videos_dir, file)
                    video_faces = self.detect_video(video_path)
                    videos_list.append(video_faces)

        videos_list = torch.stack(videos_list) / 255.0
        return videos_list


def eval_fvd(real_videos_dir: str, fake_videos_dir: str):
    fvd = FVD()
    real_videos = fvd.detect_videos(real_videos_dir)
    fake_videos = fvd.detect_videos(fake_videos_dir)
    fvd_value = compute_our_fvd(real_videos, fake_videos, device="cpu")
    print(f"FVD: {fvd_value:.3f}")


if __name__ == "__main__":
    real_videos_dir = "dir1"
    fake_videos_dir = "dir2"
    eval_fvd(real_videos_dir, fake_videos_dir)
