import time
import cv2
from src.face_detector import FaceDetector
from src import utils

def main(video_source, confidence=0.5, skip_frames=False):
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

    while True:
        ret, frame = video.read()
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
            frame = utils.draw_boxes_with_scores(frame, bboxes, scores)
            frame = utils.put_text_on_image(frame, text='FPS: {:.2f}'.format(fps_avg))

            # Show the video frame with detections
            cv2.imshow('Webcam Video', frame)

        n_frames += 1

        # Exit the loop when 'ESC' or 'q' is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):  # Press 'ESC' or 'q' to quit
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage: enable frame skipping by setting skip_frames=True
    main(video_source="video.mp4", confidence=0.5, skip_frames=True)
