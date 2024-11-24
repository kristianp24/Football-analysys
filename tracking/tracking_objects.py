from ultralytics import YOLO
import supervision as sv
import json
import os
import cv2
import pickle

class Tracker:
    def __init__(self,path):
       self.model = YOLO(path)
       self.tracker = sv.ByteTrack()

    def frame_detection(self,frames):
        detections = []
        iteration = 20

        for i in range(0,len(frames),iteration):
            detection = (self.model.predict(frames[i:i+iteration]))
            #print(detection)
            
            detections += detection
            
        return detections
      
        

    def track_objects(self, frames, path_to_json):

        if path_to_json is not None:
            with open(path_to_json, "r") as file:
                tracked_data = json.load(file)
                if tracked_data:
                    return tracked_data
        


        detected_frames = self.frame_detection(frames)

        # intializing a data structure to store the tracks
        tracked_data = {
                "player": [],
                "referees": [],
                "ball": []
            }

        for number, frame in enumerate(detected_frames):
            object_names = frame.names
            object_names_values_inverted = {v:k for k,v in object_names.items()}

            # detections with supervision
            supervision_detection = sv.Detections.from_ultralytics(frame)

            # change goalkeeper to player
            for object_nr, object_id in enumerate(supervision_detection.class_id):
                if object_id == object_names_values_inverted["goalkeeper"]:
                    supervision_detection.class_id[object_nr] = object_names_values_inverted["player"]
            
            # getting detections with track
            detectons_with_track = self.tracker.update_with_detections(supervision_detection)


           # print(detectons_with_track)
           # break

            for detection in detectons_with_track:
                bounding_box = detection[0]
                class_id = detection[3]
                tracker_id = detection[4]

                if detection[5]['class_name'] == 'player':
                    tracked_data['player'].append({
                        "bbox": bounding_box.tolist(),
                        "class_id": int(class_id),
                        "tracker_id": int(tracker_id),
                        "frame_number": number
                    })
                
                if detection[5]['class_name'] == 'referee':
                    tracked_data['referees'].append({
                        "bbox": bounding_box.tolist(),
                        "class_id": int(class_id),
                        "tracker_id": int(tracker_id),
                        "frame_number": number
                    })
                
                if detection[5]['class_name'] == 'ball':
                    tracked_data['ball'].append({
                        "bbox": bounding_box.tolist(),
                        "class_id": int(class_id),
                        "tracker_id": int(tracker_id),
                        "frame_number": number
                    })
                

        
        with open(path_to_json, "w") as data_file:
            json.dump(tracked_data,data_file, indent=4)
        
        return tracked_data
    

   
    def draw_ellipse(self,frames, tracked_data):
        output = []

        for nr, frame in enumerate(frames):
            aux_frame = frame.copy()

            player_list = [player for player in tracked_data['player'] if player['frame_number'] == nr]
            referee_list = [referee for referee in tracked_data['referees'] if referee['frame_number'] == nr]
            ball_list = [ball for ball in tracked_data['ball'] if ball['frame_number'] == nr]

            for player in player_list:
                
                    x1, y1, x2, y2 = player['bbox']
                    center_x = int((x1+x2) / 2)
                    width = x2 - x1

                    cv2.ellipse(
                        aux_frame,
                        center=(center_x,int(y2)),
                        axes=(int(width), int(0.4*width)),
                        angle = 0.0,
                        startAngle=-40,
                        endAngle=260,
                        color = (255, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                
            for referee in referee_list:
                    
                        x1, y1, x2, y2 = referee['bbox']
                        center_x = int((x1+x2) / 2)
                        width = x2 - x1

                        cv2.ellipse(
                            aux_frame,
                            center=(center_x,int(y2)),
                            axes=(int(width), int(0.4*width)),
                            angle = 0.0,
                            startAngle=-40,
                            endAngle=260,
                            color = (0, 255, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA
                        )
            
            for ball in ball_list:
                 x1, y1, x2, y2 = ball['bbox']
                 center_x = int((x1+x2) / 2)
                 width = x2 - x1
                
                 cv2.ellipse(
                            aux_frame,
                            center=(center_x,int(y2)),
                            axes=(int(width), int(0.4*width)),
                            angle = 0.0,
                            startAngle=0,
                            endAngle=360,
                            color = (0, 0, 255),
                            thickness=-1,
                            lineType=cv2.LINE_4
                        )


            output.append(aux_frame)

        return output

            
