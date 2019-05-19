import face_recognition
import glob
import os
import pickle

IMAGE_PATH = '1/'
paths = glob.glob(os.path.join(IMAGE_PATH, '*.jpg'))
paths.sort()
known_face_encodings = []
known_face_real_names = []
for i in paths:
    photo = face_recognition.load_image_file(i)
    print(i[len(IMAGE_PATH):len(i)-4])
    face_encoding = face_recognition.face_encodings(photo,num_jitters = 200)[0]
    known_face_encodings.append(face_encoding)
    known_face_real_names.append(i[len(IMAGE_PATH):len(i)-4])
print('All ' + str(len(known_face_encodings)) + ' peoples')

face_DATA = {}
for i in range(0,len(known_face_encodings)):
    face_DATA[known_face_real_names[i]] = known_face_encodings[i]
with open('faces.dat', 'wb') as f:
    pickle.dump(face_DATA, f)

print('SAVE OK')