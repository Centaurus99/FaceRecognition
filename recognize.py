import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
import math
import os
import re
import multiprocessing as mp

EPS = 0.35
Textsize = 10

def Bar(args):
    print(args)

def face_distance_to_conf(face_distance, face_match_threshold=EPS):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def Rotate(photo):
    image=Image.open(photo)
    try:
        # Grab orientation value.
        image_exif = image._getexif()
        image_orientation = image_exif[274]
        # Rotate depending on orientation.
        if image_orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if image_orientation == 3:
            image = image.transpose(Image.ROTATE_180)
        if image_orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if image_orientation == 5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        if image_orientation == 6:
            image = image.transpose(Image.ROTATE_270)
        if image_orientation == 7:
            image = image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_90)
        if image_orientation == 8:
            image = image.transpose(Image.ROTATE_90)
        if 2 <= image_orientation and image_orientation <= 8:
            print('Rotate!')
            image.save(photo, quality=95)
    except:
        pass

def Recognize(photo,all_face_encodings):
    print('Processing:' + photo)
    Rotate(photo)

    # Grab the list of names and the list of encodings
    known_face_names = list(all_face_encodings.keys())
    known_face_encodings = np.array(list(all_face_encodings.values()))

    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(photo)
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Create a Pillow ImageDraw Draw instance to draw with
    pil_image = Image.fromarray(unknown_image)
    pil_image_copy = pil_image.copy()
    draw = ImageDraw.Draw(pil_image)

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
    res_image = Image.blend(pil_image_copy, pil_image, 0.8)
    res_image.save(photo[0:len(photo)-4] + "_with_boxes.jpg", quality=95)

    print('-----')
    print('OK ' + str(len(face_locations)) + ' faces')
    print(res_image.mode,res_image.size)
    return '--------' + photo


if __name__ =='__main__':
    po = mp.Pool()
    mgr = mp.Manager()
    # Load face encodings
    with open('faces.dat', 'rb') as f:
        data_in = pickle.load(f)
    all_face_encodings = mgr.dict()
    for key in data_in:
        all_face_encodings[key] = data_in[key]

    print('All ' + str(len(all_face_encodings)) + ' peoples')

    IMAGE_PATH = '2/'
    paths = []
    for dirpath, dirnames, filenames in os.walk(IMAGE_PATH):
        for filepath in filenames:
            image_name=filepath.lower()
            match_obj = re.match(r'(?!.*with_boxes\.jpg)(.*\.jpg)',image_name)
            if match_obj:
                paths.append(os.path.join(dirpath, image_name))

    paths.sort()
    print('All ' + str(len(paths)) + ' photos')

    for photo in paths:
        po.apply_async(func = Recognize, args = (photo,all_face_encodings),callback = Bar)
    po.close()
    po.join()