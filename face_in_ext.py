import face_recognition
import glob
import os
import pickle
import multiprocessing as mp

IMAGE_PATH = 'data/'

def ImageIn(face_DATA, i):
    photo = face_recognition.load_image_file(i)
    name = i[len(IMAGE_PATH):len(i)-4]
    face_locations = face_recognition.face_locations(photo)
    if (len(face_locations) != 1):
        return 'The number of faces is ' + str(len(face_locations)) + ' but not 1'
    face_encoding = face_recognition.face_encodings(photo, face_locations, num_jitters = 200)[0]
    face_DATA[name] = face_encoding
    return name

def Bar(args):
    print(args)

if __name__ =='__main__':
    paths = glob.glob(os.path.join(IMAGE_PATH, '*.jpg'))
    paths.sort()
    
    po = mp.Pool()
    mgr = mp.Manager()
    face_DATA = mgr.dict()
    with open('faces.dat', 'rb') as f:
        data_in = pickle.load(f)
    for key in data_in:
        face_DATA[key] = data_in[key]
    print('Start!')
    for i in paths:
        po.apply_async(func = ImageIn, args = (face_DATA, i),callback = Bar)
    po.close()
    po.join()

    print('All ' + str(len(face_DATA)) + ' peoples')
    out_Data = face_DATA.copy()
    with open('faces.dat', 'wb') as f:
        pickle.dump(out_Data, f)
    print('SAVE OK')