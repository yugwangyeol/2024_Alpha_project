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
    print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")

def process_videos(input_dir):
    for phase in ['training_videos', 'testing_videos']:
        phase_dir = os.path.join(input_dir, phase)
        
        if not os.path.exists(phase_dir):
            print(f"Directory {phase_dir} does not exist. Skipping.")
            continue

        for video_file in os.listdir(phase_dir):
            if video_file.endswith('.avi'):
                video_path = os.path.join(phase_dir, video_file)
                video_name = os.path.splitext(video_file)[0]
                output_dir = os.path.join(phase_dir, video_name)
                extract_frames(video_path, output_dir)

if __name__ == '__main__':
    input_directory = '../avenue' 
    process_videos(input_directory)
