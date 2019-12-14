import imageio
import os
import numpy as np
from tqdm import tqdm
import dlib
class vidrdf:
    def __init__(self, root,filename):
        f = os.path.join(root, filename)

        self.vid = imageio.get_reader(f,  'ffmpeg')

    def do_something(self):
        points_old = np.zeros((68, 2), dtype=np.float32)
        print("Number of frames is ", self.vid.count_frames())

        point_seq = []
        img_seq = []
        # print("Number of frames: {}".format(vid.get_length()))
        # Andrew: use vid.count_frames()
        # source: https://stackoverflow.com/questions/54778001/how-to-to-tackle-overflowerror-cannot-convert-float-infinity-to-integer
        for frm_cnt in tqdm(range(0, self.vid.count_frames())):
            points = np.zeros((68, 2), dtype=np.float32)

            try:
                img = self.vid.get_data(frm_cnt)
            except:
                print('FRAME EXCEPTION!!')
                continue

            dets = detector(img, 1)
            if len(dets) != 1:
                print('FACE DETECTION FAILED!!')
                continue

            for k, d in enumerate(dets):
                shape = predictor(img, d)

                for i in range(68):
                    points[i, 0] = shape.part(i).x
                    points[i, 1] = shape.part(i).y

            # points = np.reshape(points, (points.shape[0]*points.shape[1], ))
            point_seq.append(deepcopy(points))

'''
f = os.path.join(root, filename)
            # print("f: {}".format(f))

            vid = imageio.get_reader(f,  'ffmpeg')
            point_seq = []
            img_seq = []
            # print("Number of frames: {}".format(vid.get_length()))
            # Andrew: use vid.count_frames()
            # source: https://stackoverflow.com/questions/54778001/how-to-to-tackle-overflowerror-cannot-convert-float-infinity-to-integer
            for frm_cnt in tqdm(range(0, vid.count_frames())):
                points = np.zeros((68, 2), dtype=np.float32)

                try:
                    img = vid.get_data(frm_cnt)
                except:
                    print('FRAME EXCEPTION!!')
                    continue

                dets = detector(img, 1)
                if len(dets) != 1:
                    print('FACE DETECTION FAILED!!')
                    continue

                for k, d in enumerate(dets):
                    shape = predictor(img, d)

                    for i in range(68):
                        points[i, 0] = shape.part(i).x
                        points[i, 1] = shape.part(i).y

                # points = np.reshape(points, (points.shape[0]*points.shape[1], ))
                point_seq.append(deepcopy(points))

'''

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

vidfile = 'movie.mov'
rdfobj = vidrdf('/home/Talking-Face-Landmarks-from-Speech/obama-1-dataset','16_sec_obama.mp4')
rdfobj.do_something()