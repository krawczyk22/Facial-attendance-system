# importing necessary libraries
import numpy as np
import cv2
import sys
import os
import mysql.connector

RESIZE_FACTOR = 4

class RecogPCAandLDA:
    def __init__(self):
        haarcascadePath = "haarcascade_frontalface_default.xml"
        self.haarcascade = cv2.CascadeClassifier(haarcascadePath)
        self.face_dir = 'face_data'
        self.classroom_code = sys.argv[1]
        self.modelLDA = cv2.face.FisherFaceRecognizer_create()
        self.modelPCA = cv2.face.EigenFaceRecognizer_create()
        self.face_names = []

    # loading the trained algorithms
    def load_trained_PCA_LDA(self):
        names = {}
        key = 0

        # Reading the trained LDA and PDA data in the current directories
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names 
        self.modelLDA.read('LDA_data.xml')
        self.modelPCA.read('PCA_data.xml')

    # initialising the camera
    def show_video(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            inImg = np.array(frame)
            outImg, self.face_names = self.process_images(inImg)
            cv2.imshow('Video', outImg)

            # When everything is finished, press 'q' to close computing the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                return

    # function updating the database
    def update_database(self, person):
        # defining the database connection
        mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="password",
        database="project",
        auth_plugin="mysql_native_password"
        )

        mycursor = mydb.cursor()
        sql = "INSERT INTO attendance (studentID, classroom, date_time) VALUES (%s, %s, NOW());"
        val = (person, self.classroom_code)
        mycursor.execute(sql, val)
        mydb.commit()

    # function extracing the facial features and resizing the derived image
    def process_images(self, inImg):
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (92, 112) # Fixing the resolution of the face images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Changing the colours of the images into the grey scale       
        gray_resized = cv2.resize(gray, (int(round(gray.shape[1]/RESIZE_FACTOR)), int(round(gray.shape[0]/RESIZE_FACTOR))))        
        faces = self.haarcascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
        people = []
        for i in range(len(faces)):
            face_i = faces[i]
            x = face_i[0] * RESIZE_FACTOR
            y = face_i[1] * RESIZE_FACTOR
            w = face_i[2] * RESIZE_FACTOR
            h = face_i[3] * RESIZE_FACTOR
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (resized_width, resized_height))

            # Assigning the similarity level (the lower the level is, the more similar the face picture is)
            confidenceLDA = self.modelLDA.predict(face_resized)
            confidencePCA = self.modelPCA.predict(face_resized)

            #If the similarity level of LDA and PCA meet the treshold, show the appropriate recognition rate with the recognised person's ID
            if confidenceLDA[1]<300 and confidencePCA[1]<3500: 
                personLDA = self.names[confidenceLDA[0]]
                personPCA = self.names[confidencePCA[0]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(frame, 'LDA: ' + personLDA + ' - ' + str(round(confidenceLDA[1])), (x-10, y-25), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                cv2.putText(frame, 'PCA: ' + personPCA + ' - ' + str(round(confidencePCA[1])), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            elif confidenceLDA[1]<300:
                personLDA = self.names[confidenceLDA[0]]
                personPCA = 'Unknown'
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(frame, 'LDA: ' + personLDA + ' - ' + str(round(confidenceLDA[1])), (x-10, y-25), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                cv2.putText(frame, 'PCA: ' + personPCA, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
            elif confidencePCA[1]<3500:
                personLDA = 'Unknown'
                personPCA = self.names[confidencePCA[0]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(frame, 'LDA: ' + personLDA, (x-10, y-25), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
                cv2.putText(frame, 'PCA: ' + personPCA + ' - ' + str(round(confidencePCA[1])), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
                personLDA = 'Unknown'
                personPCA = 'Unknown'
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, 'LDA: ' + personLDA, (x-10, y-25), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
                cv2.putText(frame, 'PCA: ' + personPCA, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
            people.append(personLDA)
            people.append(personPCA)

            # if a person is recognised, update the database
            if str(personLDA) != 'Unknown':
                self.update_database(personLDA)
            elif str(personPCA) != 'Unknown':
                self.update_database(personPCA)
        return (frame, people)

# callig the main function
if __name__ == '__main__':
    recogniserImage = RecogPCAandLDA()
    recogniserImage.load_trained_PCA_LDA()
    print("Press 'q' to quit")
    recogniserImage.show_video()