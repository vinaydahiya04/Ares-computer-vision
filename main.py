import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

class MonoVideoOdometery(object):
    def __init__(self, 
                img_file_path,
                pose_file_path,
                focal_length = 718.8560,
                pp = (607.1928, 185.2157), 
                lk_params=dict(winSize  = (21,21), criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)), 
                detector=cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
       

        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0

        try:
            if not all([".jpg" in x for x in os.listdir(img_file_path)]):
                raise ValueError("img_file_path is not correct and does not exclusively png files")
        except Exception as e:
            print(e)
            raise ValueError("The designated img_file_path does not exist, please check the path and try again")

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.process_frame()


    def hasNextFrame(self):
        

        return self.id < len(os.listdir(self.file_path)) 


    def detect(self, img):
      
       
        p0 = self.detector.detect(img)
        
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


    def visual_odometery(self):
        

        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)


   
        self.p1, st, err = cv.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
   
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        if self.id < 2:
            E, _ = cv.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv.RANSAC, 0.999, 1.0, None)
            _, self.R, self.t, _ = cv.recoverPose(E, self.good_old, self.good_new, self.R, self.t, self.focal, self.pp, None)
        else:
            E, _ = cv.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv.RANSAC, 0.999, 1.0, None)
            _, R, t, _ = cv.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), self.focal, self.pp, None)

            absolute_scale = self.get_absolute_scale()
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t = self.t + absolute_scale*self.R.dot(t)
                self.R = R.dot(self.R)

        self.n_features = self.good_new.shape[0]


    def get_mono_coordinates(self):
 
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()

    def get_absolute_scale(self):
        
        print(self.id)
        pose = self.pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])


        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
        
        return np.linalg.norm(true_vect - prev_vect)


    def process_frame(self):
       

        if self.id < 2:
            
            self.old_frame = cv.imread(self.file_path +str().zfill(5)+'.jpg', 0)
            self.current_frame = cv.imread(self.file_path + str(1).zfill(5)+'.jpg', 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv.imread(self.file_path + str(self.id).zfill(5)+'.jpg', 0)
            self.visual_odometery()
            self.id += 1


img_path = 'E:\\2.0\\Projects\\ares\\Deliverable\\CV\\sequence_01\\'
pose_path = 'E:\\2.0\\Projects\\ares\\Deliverable\\CV\\extra\\groundtruthSync.txt'


vo = MonoVideoOdometery(img_path, pose_path)

while(vo.hasNextFrame()):

    frame = vo.current_frame

    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break

    vo.process_frame()  
    mono_coord = vo.get_mono_coordinates()  
    print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))   

cv.destroyAllWindows()

