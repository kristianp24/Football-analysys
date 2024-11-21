from video_utils import VideoUtils
from tracking import Tracker as trk

def main():
    video_frames = VideoUtils.readVideo('input/8fd33_4.mp4')

    tracker = trk('colab_training/best.pt')
    tracked_data = tracker.track_objects(frames = video_frames, path_to_json='saved_data/tracked_data.json')
    VideoUtils.writeVideo(video_frames,'writed_video/writed_from_function.avi')



if __name__ == "__main__":
    main()
