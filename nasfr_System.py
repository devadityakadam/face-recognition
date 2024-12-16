import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np


window=tk.Tk()
window.title("nasfr_system")

l1=tk.Label(window,text="Name",font=("Algerian",20))
l1.grid(column=0,row=0)
t1=tk.Entry(window,width=50,bd=5)
t1.grid(column=1,row=0)

l2=tk.Label(window,text="Age",font=("Algerian",20))
l2.grid(column=0,row=1)
t2=tk.Entry(window,width=50,bd=5)
t2.grid(column=1,row=1)

l3=tk.Label(window,text="Address",font=("Algerian",20))
l3.grid(column=0,row=2)
t3=tk.Entry(window,width=50,bd=5)
t3.grid(column=1,row=2)

def train_classifier():
    data_dir="C:/Users/DELL/OneDrive/Desktop/face recognization/images"
    path=[os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces=[]
    ids=[]

    for image in path:
        img =Image.open(image).convert('L')
        imageNp=np.array(img,'uint8')
        id=int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids=np.array(ids)

    #train the classifier and save
    clf =cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)

    clf.write("classifier.xml")
    messagebox.showinfo('result','training dataset is compleated')

b1=tk.Button(window,text="Training",font=("Algerian",20),bg='gray',fg='black',command=train_classifier)
b1.grid(column=0,row=3)


def generate_dataset():
    if(t1.get()=="" or t2.get()=="" or t3.get()==""):
       messagebox.showinfo('Result','Please provide complete details of the user')
    else:
        face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        name= input("enter id:")

        def face_cropped(img):
            grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(grey,1.3,5)


            if faces is ():
                return None
            for (x,y,w,h) in faces:
                cropped_face=img[y:y+h,x:x+w]
            return cropped_face

        cap=cv2.VideoCapture(0)
        id=name
        img_id=0
        if not os.path.exists('images'):
            path = os.makedirs('images')
        while True:
            ret,frame=cap.read()
            # if not os.path.exists('images/' +name):
            #     path = os.makedirs('images/' +name)
            if face_cropped(frame) is not None:
                img_id+=1
                face = cv2.resize(face_cropped(frame),(200,200))
                face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                file_name_path="images/user."+str(id)+"."+str(img_id)+".jpg"
                cv2.imwrite(file_name_path,face)
                cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

                cv2.imshow("Cropped face",face)
                if cv2.waitKey(1)==13 or int(img_id)==300:
                    break


        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('result','collecting samples is completed.........')

b2=tk.Button(window,text="Ganerate Dataset",font=("Algerian",20),bg='gray',fg='black',command=generate_dataset)
b2.grid(column=1,row=3)

def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fratures = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in fratures:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence >80:
                if id == 1:
                    cv2.putText(img, "Aditya kadam", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id == 2:
                    cv2.putText(img, "Nehal", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id == 3:
                    cv2.putText(img, "Kundan bondre", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "unknown person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1,
                            cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()


b3=tk.Button(window,text="Detect Face",font=("Algerian",20),bg='gray',fg='black',command=detect_face)
b3.grid(column=2,row=3)


window.geometry("800x200")
window.mainloop()