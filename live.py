import face_recognition
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pickle
import time
import os
import multiprocessing as mp

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

EPS = 0.35
Textsize = 10

def face_distance_to_conf(face_distance, face_match_threshold=EPS):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def Recognize(unknown_image, all_face_encodings, res):
    
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    print('t1')
    # Grab the list of names and the list of encodings
    known_face_names = list(all_face_encodings.keys())
    known_face_encodings = np.array(list(all_face_encodings.values()))

    print('t2')
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    print('t3')
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
        res.append(((top, right, bottom, left),name))
    print('t4')


if __name__ =='__main__':
    mgr = mp.Manager()
    # Load face encodings
    with open('faces.dat', 'rb') as f:
        data_in = pickle.load(f)
    all_face_encodings = mgr.dict()
    for key in data_in:
        all_face_encodings[key] = data_in[key]

    print('All ' + str(len(all_face_encodings)) + ' peoples')

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Initialize some variables

    CPU_NUM = max(mp.cpu_count() - 1, 1)
    which = []
    res = []
    is_run = []
    process = []
    pic = []

    for i in range(0,CPU_NUM):
        which.append(None)
        res.append(None)
        is_run.append(False)
        process.append(None)
        pic.append(None)
        
    nowT = 0
    mxT = 0
    faces_data = []
    while True:
        nowT += 1
        # Grab a single frame of video
        ret, frame = video_capture.read()
        tag = -1
        
        for i in range(0,CPU_NUM):
            if is_run[i] == False:
                tag = i
                continue
            if process[i].is_alive() == False:
                is_run[i] = False
                tag = i
                if which[i] > mxT:
                    mxT = which[i]
                    faces_data = res[i]
        print('-------' + str(nowT))
        print(which)
        print(res)
        print(is_run)
        print(process)
        print(tag)
        
        if tag != -1:
            which[tag] = nowT
            res[tag] = mgr.list()
            is_run[tag] = True
            process[tag] = mp.Process(target = Recognize, args = (all_face_encodings, res[tag]))
            process[tag].start()
        
        
        print(faces_data)
        pil_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        pil_image_copy = pil_image.copy()
        draw = ImageDraw.Draw(pil_image)
        # Display the results
        for (top, right, bottom, left), name in faces_data:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

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

        #print('OK ' + str(len(face_locations)) + ' faces')

        # Remove the drawing library from memory as per the Pillow docs
        del draw
        res_image = Image.blend(pil_image_copy, pil_image, 0.7)
        #print(res_image.mode,res_image.size)

        frame = cv2.cvtColor(np.asarray(res_image),cv2.COLOR_BGR2RGB)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()