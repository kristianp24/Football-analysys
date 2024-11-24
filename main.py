from video_utils import VideoUtils
from tracking import Tracker as trk
import pickle
import cv2

def main():
    video_frames = VideoUtils.readVideo('input/8fd33_4.mp4')
    
   

    tracker = trk('colab_training/best.pt')

    tracked_data = tracker.track_objects(frames = video_frames, path_to_json='saved_data/tracked_data.json')
    print(tracked_data['ball'])

    #players = [player for player in tracked_data['player'] if player['frame_number'] == 0]
    # for player in players:
    #         frame = video_frames[0]
    #         x1, y1, x2, y2 = player['bbox']
    #         image = frame[int(y1):int(y2), int(x1):int(x2)]
            
    #         cv2.imwrite(f'image_for_clustering/player.jpg', image)
    #         break
        
    # drawed_frames = tracker.draw_ellipse(video_frames, tracked_data)
    # VideoUtils.writeVideo(drawed_frames,'writed_video/output_ellipses2.avi')



if __name__ == "__main__":
    main()
