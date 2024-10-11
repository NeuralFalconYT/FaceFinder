import os
import time
import uuid  # Import UUID for generating random filenames
import cv2
from src.face_detector import FaceDetector
from src import utils

def main(video_source, confidence=0.5, play_fast=True, skip_frames=False, display_fps=False, bounding_box=True, save_faces=False, circle_blur_face=False, square_blur_face=False, save_video=False):
    # Initialize the face detector with the specified model
    detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=confidence,
                            overlap_thr=0.7)

    # Open the video source (0 for the default webcam or specify a video file)
    video = cv2.VideoCapture(video_source)

    # Automatically get FPS from the video source
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps) if skip_frames else 1  # Set frame skip based on FPS or process every frame

    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0

    # Check if saving video is enabled
    if save_video:
        # Create a directory for saving videos if it doesn't exist
        video_dir = "videos"  # Directory to save videos
        os.makedirs(video_dir, exist_ok=True)  # Create the directory if it doesn't exist
        if video_source not in [0, 1, 2, 3,4,5,6,7,8,9]:
            video_filename = os.path.splitext(os.path.basename(video_source))[0]
            # Generate a random UUID for the video filename
            video_filename = f"{video_dir}/{video_filename}_{str(uuid.uuid4())[:6]}.mp4"
        else:
            # Generate a random UUID for the video filename
            video_filename = f"{video_dir}/{str(uuid.uuid4())[:6]}.mp4"  # Random UUID filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        out = cv2.VideoWriter(video_filename, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))  # Initialize VideoWriter

    while True:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Error reading frame from the video source.")
            break

        # Skip frames based on the frame skip setting
        if n_frames % frame_skip == 0:
            start_time = time.perf_counter()
            bboxes, scores = detector.inference(frame)
            end_time = time.perf_counter()

            fps = 1.0 / (end_time - start_time)
            fps_cum += fps
            fps_avg = fps_cum / (n_frames // frame_skip + 1)

            # Draw bounding boxes and display FPS
            frame = utils.draw_boxes_with_scores(frame, bboxes, scores, bounding_box=bounding_box,
                                                  save=save_faces, circle_blur_face=circle_blur_face,
                                                  square_blur_face=square_blur_face)
            if display_fps:
                frame = utils.put_text_on_image(frame, text='FPS: {:.2f}'.format(fps_avg))

            # Show the video frame with detections
            cv2.imshow('Webcam Video', frame)

            # Save the frame to the video file if save_video is enabled
            if save_video:
                out.write(frame)

            if not play_fast:
                # Wait for a specific time based on the video's FPS
                wait_time = int(1000 / fps)  # Convert FPS to milliseconds
                cv2.waitKey(wait_time)  # Wait for the calculated time

        n_frames += 1

        # Exit the loop when 'ESC' or 'q' is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):  # Press 'ESC' or 'q' to quit
            break

    # Release resources
    video.release()
    if save_video:
        out.release()  # Release the VideoWriter if saving video
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # video_source = 0
    video_source = "video.mp4"  # Path to the video file
    confidence = 0.5
    play_fast = True
    skip_frames = False
    display_fps = False
    bounding_box = False
    save_faces = False
    circle_blur_face = False
    square_blur_face = True
    save_video = True  # Set to True to save video

    main(video_source=video_source, confidence=confidence, play_fast=play_fast, skip_frames=skip_frames,
         display_fps=display_fps, bounding_box=bounding_box, save_faces=save_faces,
         circle_blur_face=circle_blur_face, square_blur_face=square_blur_face, save_video=save_video)
