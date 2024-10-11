import os
import cv2

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    frame_count = 0

    while success:
        frame_filename = os.path.join(output_dir, "{:04d}.jpg".format(frame_count))
        cv2.imwrite(frame_filename, frame)
        success, frame = video_capture.read()
        frame_count += 1

    video_capture.release()
    print(f"Extracted {frame_count} framㅁes from {video_path} to {output_dir}")

def process_videos(video_dir, frames_dir):
    if not os.path.exists(video_dir):
        print(f"Directory {video_dir} does not exist.")
        return

    for video_file in os.listdir(video_dir):
        if video_file.endswith('.avi'):
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_dir = os.path.join(frames_dir, video_name)
            extract_frames(video_path, output_dir)

if __name__ == '__main__':
    video_directory = '/home/work/Alpha/Jigsaw-VAD/shanghaitech/training/videos'
    frames_directory = '/home/work/Alpha/Jigsaw-VAD/shanghaitech/training/frames'
    process_videos(video_directory, frames_directory)
