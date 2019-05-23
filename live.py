import face_recognition
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pickle
import time
import os
import multiprocessing as mp
import math

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

EPS = 0.40
Textsize = 10
Rate = 1
RecoInterval = 0.5
EXT_PHOTO_PATH = 'data/'

def cv_imwrite(out_path, img_np):
    cv2.imencode('.jpg', img_np, [int(cv2.IMWRITE_JPEG_QUALITY),95])[1].tofile(EXT_PHOTO_PATH + out_path)

def face_distance_to_conf(face_distance, face_match_threshold=EPS):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def Recognize(all_face_encodings, res, is_run, pic, wh):
    while True:
        while is_run[wh] == False:
            time.sleep(0.01)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = pic[wh][:, :, ::-1]

        # Grab the list of names and the list of encodings
        known_face_names = list(all_face_encodings.keys())
        known_face_encodings = np.array(list(all_face_encodings.values()))

        #t1 = time.process_time()

        # Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(rgb_small_frame)

        #t2 = time.process_time() - t1
        #print(str(wh) + '-1:' + str(t2))
        
        #t1 = time.process_time()

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        #t2 = time.process_time() - t1
        #print(str(wh) + '-2:' + str(t2))

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            name = 'Unknown'

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < EPS:
                name = known_face_names[best_match_index]
            if name == 'Unknown':
                name += '(' + str(round(face_distance_to_conf(face_distances[best_match_index])*100,1)) + '%' + known_face_names[best_match_index] + ')'
            else:
                name += '(' + str(round(face_distance_to_conf(face_distances[best_match_index])*100,1)) + '%)'
            res[wh].append(((top, right, bottom, left),name))
        is_run[wh] = False


if __name__ =='__main__':
    
    if not os.path.isdir(EXT_PHOTO_PATH):
        os.makedirs(EXT_PHOTO_PATH)

    mgr = mp.Manager()
    # Load face encodings
    with open('faces.dat', 'rb') as f:
        data_in = pickle.load(f)
    all_face_encodings = mgr.dict()
    for key in data_in:
        all_face_encodings[key] = data_in[key]

    print('All ' + str(len(all_face_encodings)) + ' peoples')

    # Initialize some variables

    CPU_NUM = max(mp.cpu_count() - 1, 1)
    which = []
    res = mgr.list()
    is_run = mgr.list()
    process = []
    pic = mgr.list()

    for i in range(0,CPU_NUM):
        which.append(0)
        res.append(mgr.list())
        pic.append(None)
        is_run.append(False)
        process.append(mp.Process(target = Recognize, args = (all_face_encodings, res, is_run, pic, i)))
        process[i].daemon = True
        process[i].start()
    time.sleep(2)

    nowT = 0
    mxT = 0
    faces_data = []
    
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    fps =video_capture.get(cv2.CAP_PROP_FPS)
    print('fps: ' + str(fps))
    Inversal = math.floor(fps*RecoInterval)
    print('Inversal: ' + str(Inversal))
    fps = math.floor(fps)
    run_sums = 0
    while True:
        nowT += 1
        # Grab a single frame of video
        ret, frame = video_capture.read()
        tag = -1
        for i in range(0,CPU_NUM):
            if is_run[i] == False:
                tag = i
                if which[i] > mxT:
                    mxT = which[i]
                    faces_data = res[i]
                    print('[faces: ' + str(len(faces_data)) + ', delay: ' + str(nowT-which[i]) + ']')
                which[i] = 0
            else:
                run_sums += 1
        
        if nowT % (fps * 5) == 0:
            print('----------Queueing: ' + str(round(run_sums / (fps * 5), 2)) + '/' + str(CPU_NUM))
            run_sums = 0

        '''
        print('-------' + str(nowT))
        print(which)
        print(res)
        print(is_run)
        '''

        if nowT%Inversal == 0 and tag != -1:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=1/Rate, fy=1/Rate)
            pic[tag] = small_frame
            which[tag] = nowT
            res[tag] = mgr.list()
            is_run[tag] = True
        
        
        pil_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        pil_image_copy = pil_image.copy()
        draw = ImageDraw.Draw(pil_image)
        # Display the results
        for (top, right, bottom, left), name in faces_data:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= Rate
            right *= Rate
            bottom *= Rate
            left *= Rate

            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(40, 42, 54))
            w = right - left
            h = bottom - top
            nowsize = Textsize
            
            while (1):
                now_w, now_h = draw.textsize(name,font=ImageFont.truetype('msyh.ttc',nowsize+1))
                if (now_w > w or now_h * 8 > h):
                    break
                nowsize += 1
            
            font = ImageFont.truetype('msyh.ttc',nowsize)
            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name, font=font)
            if left + text_width + 6 > right:
                left -= (left + text_width + 6 - right) // 2
            draw.rectangle(((left, bottom - text_height - 6), (max(right, left + text_width + 6), bottom)), fill=(40, 42, 54), outline=(40, 42, 54))
            draw.text((left + 3, bottom - text_height - 3), name, fill=(255, 255, 255), font=font)

        # Remove the drawing library from memory as per the Pillow docs
        del draw
        res_image = Image.blend(pil_image_copy, pil_image, 0.7)

        key = cv2.waitKey(1) & 0xFF
        if  key == ord('p'):
            for (top, right, bottom, left), name in faces_data:
                top *= Rate
                right *= Rate
                bottom *= Rate
                left *= Rate
                top = max(0, top - 20)
                left = max(0, left - 20)
                bottom = min(frame.shape[0], bottom + 20)
                right = min(frame.shape[1], right + 20)
                pht = frame[top:bottom, left:right]
                cv_imwrite(name + '.jpg', pht)
                print(name + '.jpg saved')
        
        frame = cv2.cvtColor(np.asarray(res_image),cv2.COLOR_BGR2RGB)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if  key == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()