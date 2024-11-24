from video_utils import VideoUtils
from tracking import Tracker as trk
import pickle

def main():
    video_frames = VideoUtils.readVideo('input/8fd33_4.mp4')
    
   

    tracker = trk('colab_training/best.pt')

    tracked_data = tracker.track_objects(frames = video_frames, path_to_json='saved_data/tracked_data.json')
    # print(tracked_data['player'], type(tracked_data['player']))
    # for player in tracked_data['player']:
    #    if player['frame_number'] == 0:
    #        print(player)
    drawed_frames = tracker.draw_ellipse(video_frames, tracked_data)
    VideoUtils.writeVideo(drawed_frames,'writed_video/output_ellipses.avi')



if __name__ == "__main__":
    main()
