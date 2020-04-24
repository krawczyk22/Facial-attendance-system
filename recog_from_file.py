# used in recognising pictures from the file rather than a live stream
# importing necessary libraries
import numpy as np
import cv2
import sys
import os

RESIZE_FACTOR = 4

class RecogPCAandLDA:
    def __init__(self):
        haarcascadePath = "haarcascade_frontalface_default.xml"
        self.haarcascade = cv2.CascadeClassifier(haarcascadePath)
        self.face_dir = 'face_test'
        #self.classroom_code = sys.argv[1]
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

    def show_video(self):
        TLDAPCAcount = 0
        TLDAcount = 0
        TPCAcount = 0
        Tnone = 0
        falseLDA = 0
        falsePCA = 0
        confLDA = 1500
        confPCA = 1500
        listPCA =[]
        listLDA = []
        listAll = []
        print("Iteration\tLDA\tPCA\tFusion\tFalseLDA\tFalsePCA")
        for i in range(1500, 15000, 500):
            for (subdirs, dirs, files) in os.walk(self.face_dir):
                for subdir in dirs:
                    img_path = os.path.join(self.face_dir, subdir)
                    LDAPCAcount = 0
                    LDAcount = 0
                    PCAcount = 0
                    none = 0
                    for fn in os.listdir(img_path):
                        path = img_path + '/' + fn
                        listAll.append(path)
                        img = cv2.imread(path)
                        inImg = np.array(img)
                        LDAPCAcount, LDAcount, PCAcount, none, falseLDA, falsePCA, listLDA, listPCA = self.process_images(inImg, confLDA, confPCA, LDAPCAcount, LDAcount, PCAcount, none, subdir, falseLDA, falsePCA, listLDA, listPCA, path)
                    TLDAPCAcount += LDAPCAcount
                    TLDAcount += LDAcount
                    TPCAcount += PCAcount
                    Tnone += none
            #print("\nIteration " + str(confLDA) + " " + str(confPCA) + "\n")
            #print("LDA and PCA: " + str(TLDAPCAcount))
            #print("LDA: " + str((TLDAcount*100/len(listAll)-falseLDA*100/len(listAll))))
            #print("PCA: " + str((TPCAcount*100/len(listAll)-falsePCA*100/len(listAll))))
            #print("Fusion: " + str(len(list(set().union(listLDA, listPCA)))*100/len(listAll)))
            #print("None: " + str(Tnone) + "\n")
            #print("MISRECOGNITIONS")
            #print("False LDA: " + str(falseLDA*100/len(listAll)))
            #print("False PCA: " + str(falsePCA*100/len(listAll)))
            FLDA = falseLDA*100/len(listAll)
            FPCA = falsePCA*100/len(listAll)
            LDA = TLDAcount*100/len(listAll)-FLDA
            PCA = TPCAcount*100/len(listAll)-FPCA
            Fusion = len(list(set().union(listLDA, listPCA)))*100/len(listAll)
            print(str(i)+"\t\t"+str(round(LDA, 2))+"\t"+str(round(PCA, 2))+"\t"+str(round(Fusion, 2))+"\t"+str(round(FLDA, 2))+"\t\t"+str(round(FPCA, 2)))
            #print(round(Fusion, 2))
            TLDAPCAcount = 0
            TLDAcount = 0
            TPCAcount = 0
            Tnone = 0
            falseLDA = 0
            falsePCA = 0
            confPCA += 500
            confLDA += 500
            listLDA.clear()
            listPCA.clear()
            listAll.clear()

    def process_images(self, frame, confLDA, confPCA, LDAPCAcount, LDAcount, PCAcount, none, subdir, falseLDA, falsePCA, listLDA, listPCA, path):
        resized_width, resized_height = (168, 192)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Changing the colours of the images into the grey scale       
        gray_resized = cv2.resize(gray, (int(round(gray.shape[1]/RESIZE_FACTOR)), int(round(gray.shape[0]/RESIZE_FACTOR)))) 
        face_resized = cv2.resize(gray_resized, (resized_width, resized_height))
        # Assigning the similarity level (the lower the level is, the more similar the face picture is)
        confidenceLDA = self.modelLDA.predict(face_resized)
        confidencePCA = self.modelPCA.predict(face_resized)
        #If the similarity level of LDA and PCA meet the treshold, show the appropriate recognition rate with the recognised person's ID
        if confidenceLDA[1]<confLDA and confidencePCA[1]<confPCA: 
            personLDA = self.names[confidenceLDA[0]]
            personPCA = self.names[confidencePCA[0]]
            #LDAPCAcount += 1
            LDAcount += 1
            PCAcount += 1
            if str(subdir) != str(personLDA):
                falseLDA += 1
            else:
                listLDA.append(str(path))
            if str(subdir) != str(personPCA):
                falsePCA += 1  
            else:
                listPCA.append(str(path))
        elif confidenceLDA[1]<confLDA:
            personLDA = self.names[confidenceLDA[0]]
            personPCA = 'Unknown'
            LDAcount += 1
            if str(subdir) != str(personLDA):
                falseLDA += 1
            else:
                listLDA.append(str(path))
        elif confidencePCA[1]<confPCA:
            personLDA = 'Unknown'
            personPCA = self.names[confidencePCA[0]]
            PCAcount += 1
            if str(subdir) != str(personPCA):
                falsePCA += 1  
            else:
                listPCA.append(str(path))
        else:
            personLDA = 'Unknown'
            personPCA = 'Unknown'
            none += 1
        return (LDAPCAcount, LDAcount, PCAcount, none, falseLDA, falsePCA, listLDA, listPCA)

# callig the main function
if __name__ == '__main__':
    recogniserImage = RecogPCAandLDA()
    recogniserImage.load_trained_PCA_LDA()
    recogniserImage.show_video()