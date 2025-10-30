
import cv2
import numpy as np

def frame_energy(gray_frame: np.ndarray) -> float:
    """
    Compute an energy score for a frame.
    Here: mean squared intensity (basically power of the signal).
    Higher = brighter / more active content.
    """
    # gray_frame is 2D [H,W], uint8 0-255
    # convert to float so math isn't clipped
    f = gray_frame.astype(np.float32)
    return float(np.mean(f * f))


def preprocess_video(video_path, frame_step=45, energy_threshold=2000.0):
    """
    Extract subsampled frames from the video, but only keep
    frames whose energy is "high enough".
    
    Args:
        video_path: path to video file
        frame_step: how many frames to skip forward each iteration
        energy_threshold: minimum energy required to keep a frame
    
    Returns:
        frames_kept: list of resized (175x175) BGR frames that passed the filter
        energies: list of energies for debugging/analysis
    """
    cap = cv2.VideoCapture(video_path)
    frames_kept = []
    energies = []

    if not cap.isOpened():
        print("Could not open video:", video_path)
        return frames_kept, energies

    # We'll manually hop through the timeline using CAP_PROP_POS_FRAMES,
    # same logic you started.
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_idx < total_frames:
        # move to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # compute energy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        e = frame_energy(gray)

        # check if this frame has enough "energy"
        if e >= energy_threshold:
            # resize and keep
            resized = cv2.resize(frame, (175, 175))
            frames_kept.append(resized)
            energies.append(e)

        # advance to next sample position
        frame_idx += frame_step

    cap.release()
    return frames_kept
