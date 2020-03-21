from tkinter import *
import os

root = Tk(className = ' PCA LDA recognition GUI')
idnumberstring = StringVar() # defining a string for the name
classroomstring = StringVar() # defining a string for the classroom

w = Entry(root,textvariable=idnumberstring) # textarea for individual's ID No.
w.pack()

# defining the functions which contain commands used for running the appropriate parts of the program

def train_PCA_LDA_btn_load():
    name = idnumberstring.get()
    os.system('python3 train_PCA_LDA.py %s'%name)

def recog_PCA_LDA_btn_load():
    classroom = classroomstring.get()
    os.system('python3 recog_PCA_LDA.py %s'%classroom)

# executing the function buttons

trainF_btn = Button(root,text="Add student", command=train_PCA_LDA_btn_load)
trainF_btn.pack()

q = Entry(root,textvariable=classroomstring) # textarea for the classroom
q.pack()

recogF_btn = Button(root,text="Verify attendance", command=recog_PCA_LDA_btn_load)
recogF_btn.pack()

root.mainloop()

