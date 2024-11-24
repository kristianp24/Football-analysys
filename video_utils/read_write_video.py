import cv2
class VideoUtils:
    def readVideo(path):
        input = cv2.VideoCapture(path)
        frames = []
        while(input.isOpened()):
            flag, frame = input.read()
            if not flag:
                break
            frames.append(frame)
        
        return frames


    def writeVideo(frames,output_path):
        codec = cv2.VideoWriter.fourcc(*'XVID')
        video = cv2.VideoWriter(output_path,codec,24,(frames[0].shape[1],frames[0].shape[0]))

        for frame in frames:
            video.write(frame)
        video.release()


