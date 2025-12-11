import cv2
import os


def extract_frames(video_path: str, outputDir: str, interval_seconds):
    """Extract frames from video every X seconds."""
    # Clean old frames
    for file in os.listdir(outputDir):
        os.remove(os.path.join(outputDir, file))

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0
    frame_num = 1

    while current_frame <= total_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = video.read()

        if not success:
            break

        frame_path = os.path.join(outputDir, f"frame_{frame_num:03d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_num += 1
        current_frame += fps * interval_seconds

    video.release()
