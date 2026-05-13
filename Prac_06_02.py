"""
Example of usage: python Prac_06_02.py video1.mp4 --start 5 --frames 45
"""

import cv2
import os
import argparse
import imageio

def run_tracker(tracker_type, video_path, start_time_sec, max_frames, start_bbox):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    
    ret, init_frame = cap.read()
    if not ret:
        print("Failed to read initial frame.")
        return

    tracker = cv2.TrackerKCF_create() if tracker_type == 'KCF' else cv2.TrackerCSRT_create()
    tracker.init(init_frame, start_bbox)
    
    output_dir = 'tracking_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gif_frames = []

    print(f"--- Tracking with {tracker_type} ---")
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        success, bbox = tracker.update(frame)
        
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv2.putText(frame, f"{tracker_type} Frame {i+1}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(rgb_frame)
        else:
            print(f"[{tracker_type}] Lost target at frame {i+1}")

    # Export the collected frames as a GIF
    if gif_frames:
        gif_path = f"{output_dir}/{tracker_type}_result.gif"
        # duration is the time per frame in seconds (0.04s = 25fps)
        imageio.mimsave(gif_path, gif_frames, duration=0.04, loop=0)
        print(f"Successfully saved: {gif_path}")

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Compare trackers and export as GIF.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--start", type=float, default=0, help="Start time (sec)")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames")
    
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not access video at that time.")
        return

    print(f"Select ROI and press ENTER.")
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
    cv2.destroyAllWindows()
    cap.release()

    for t_type in ['KCF', 'CSRT']:
        run_tracker(t_type, args.video, args.start, args.frames, bbox)

if __name__ == "__main__":
    main()