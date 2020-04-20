# the file has been developed basing on the work of Tanay Choudhary which can be found here: https://github.com/tanay-bits/cvlib/tree/master/Face%20Recognition
# importing necessary libraries
import numpy as np
import cv2
import sys
import os
import mysql.connector

FREQ_DIV = 5   # The frequency divider for capturing training images
RESIZE_FACTOR = 4
NUM_TRAINING = 100 # The number of training images the program captures

class TrainPCAandLDA:
    def __init__(self):
        haarcascadePath = "haarcascade_frontalface_default.xml"
        self.haarcascade = cv2.CascadeClassifier(haarcascadePath)
        self.face_dir = 'face_data'
        self.face_name = sys.argv[1]
        self.path = os.path.join(self.face_dir, self.face_name)
        # if the directory with the given name does not exist, create it
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.modelPCA = cv2.face.EigenFaceRecognizer_create()
        self.modelLDA = cv2.face.FisherFaceRecognizer_create()
        self.count_images = 0
        self.count_timer = 0

    # function initialising the camera and capturing
    def capture_images(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            self.count_timer += 1
            ret, frame = video_capture.read()
            inImg = np.array(frame)
            outImg = self.process_images(inImg)
            cv2.imshow('Video', outImg)

            # When everything is finished, press 'q' to close computing the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                return

    # function extracing the facial features and resizing the derived image
    def process_images(self, inImg):
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (92, 112) # Fixing the resolution of the face images

        # Capturing the images until the required number of training images is reached        
        if self.count_images < NUM_TRAINING:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Changing the colours of the images into the grey scale
            gray_resized = cv2.resize(gray, (int(round(gray.shape[1]/RESIZE_FACTOR)), int(round(gray.shape[0]/RESIZE_FACTOR))))        
            faces = self.haarcascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
            if len(faces) > 0:
                areas = []
                for (x, y, w, h) in faces: 
                    areas.append(w*h)
                max_area, idx = max([(val,idx) for idx,val in enumerate(areas)])
                face_sel = faces[idx]
            
                x = face_sel[0] * RESIZE_FACTOR
                y = face_sel[1] * RESIZE_FACTOR
                w = face_sel[2] * RESIZE_FACTOR
                h = face_sel[3] * RESIZE_FACTOR

                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (resized_width, resized_height))
                img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.path) if fn[0]!='.' ]+[0])[-1] + 1
                
                # Saving captured images with the directory of the person's ID
                if self.count_timer%FREQ_DIV == 0:
                    cv2.imwrite('%s/%s.png' % (self.path, img_no), face_resized)
                    self.count_images += 1
                # Surrounding the face with a green frame, showing the ID number and the image count
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, self.face_name + ' - ' + str(self.count_images) + '/' + str(NUM_TRAINING), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))
        elif self.count_images == NUM_TRAINING:
            print("Training PCA and LDA finished. Press 'q' to exit.")
            self.count_images += 1

        return frame           

    # Verifying the number of faces stored in the system (fullfilling the requirement of th LDA where min 2 faces must be stored)
    def number_of_faces(self):
        existingFaces = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                existingFaces += 1

        if existingFaces > 1:
            return True
        else:
            return False

    # funtion training the PCA algorithm
    def PCA_train_data(self):
        imgs = []
        tags = []
        index = 0

        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]

        self.modelPCA.train(imgs, tags)
        self.modelPCA.save('PCA_data.xml') # Saving the trained PCA data into a file
        print("PCA training completed")
        return

    # funtion training the PCA algorithm
    def LDA_train_data(self):
        imgs = []
        tags = []
        index = 0

        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]

        self.modelLDA.train(imgs, tags)
        self.modelLDA.save('LDA_data.xml') # Saving the trained LDA data into a file
        print("LDA training completed")
        return

    # function updating the database
    def update_database(self):
        # defining the database connection
        mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="password",
        database="project",
        auth_plugin="mysql_native_password"
        )

        mycursor = mydb.cursor()
        sql = "INSERT INTO student (ID) VALUES (%s);"
        val = (self.face_name,)
        mycursor.execute(sql, val)
        mydb.commit()

# callig the main function
if __name__ == '__main__':
    trainerImages = TrainPCAandLDA()
    trainerImages.capture_images()
    
    trainerImages.PCA_train_data()
    if trainerImages.number_of_faces():
        trainerImages.LDA_train_data()
    trainerImages.update_database()
    print("Press 'q' to quit")