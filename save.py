from Tkinter import *
import Tkinter as tk
import cv2, sys, os, pyttsx, time
from PIL import Image, ImageTk
import numpy
import tkMessageBox
import tkSimpleDialog
from imutils.video import WebcamVideoStream
import imutils
import threading
import speech_recognition as sr
import led

width, height = 400,300

cascade = cv2.CascadeClassifier('faces.xml')
fn_dir = '/home/pi/Opencv/recognition/database'
#recognizer = cv2.face.createFisherFaceRecognizer()
recognizer = cv2.face.createLBPHFaceRecognizer()
(im_width, im_height) = (112,92)
recognizer.load('/home/pi/Opencv/recognition/trainer.yml')

class Application:
    
    def __init__(self):
        
        self.cap = WebcamVideoStream(src=0).start()
        self.root = tk.Tk()
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.overrideredirect(0)
        self.root.geometry("%dx%d+0+0" % (w, h))
        self.root.wm_title('Face Recognizer')

        #Top and Bottom Frame
        tFrame = tk.Frame(self.root, width=360, height=240)
        tFrame.grid(row=0, column=0, padx=10, pady=50)
        bFrame = tk.Frame(self.root, width=360, height=240)
        bFrame.grid(row=1, column=0, padx=10, pady=2)
        
        #Video Frame
        self.lmain = tk.Label(tFrame)
        self.lmain.grid(row=0, column=0)
        self.details = Label(tFrame, text = 'Details: ')
        self.details.grid(row=0, column=1)
        self.det = Label(tFrame)
        self.det.grid(row=0, column=2,)
        self.vc = Label(bFrame, text =  'Say Something')
        self.vc.grid(row=1, column=0,)
        
        #Buttons
        trainButton = Button(bFrame, text = 'Train System', command=self.ftrain)
        trainButton.grid(row=0, column=0)
        quitButton = Button(bFrame, text = 'Quit', command=self.Quit)
        quitButton.grid(row=0, column=1)
        
        
        self.video_Loop()
 
    def camera(self):
        frame = self.cap.read()
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, gray
    
    def face_det(self, frame, gray):
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        stx = 0
        sty = 0
        wdt = 0
        ht = 0
        for(x,y,w,h) in faces:
            stx = x
            sty = y
            wdt = w
            ht = h   
        cv2.rectangle(frame, (stx,sty), (stx+wdt,sty+ht), (0,0,255), 2)
        return faces
        
        
    def face_rec(self, frame, faces, gray):
        # Part 2: Use fisherRecognizer on camera stream
        (images, lables, names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(fn_dir, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    lable = id
                    images.append(cv2.imread(path, 0))
                    lables.append(int(lable)) 
                id += 1
        # Try to recognize the face
        
        for (x,y,w,h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            prediction = recognizer.predict(face_resize)
            if prediction<100:
                name = names[prediction]
                try:
                    f = open('/home/pi/Opencv/recognition/database/%s.txt'%name)
                    self.det.configure(text = f.read(), anchor = 'n', justify = 'left')
                except:
                    self.det.configure(text = 'No Information')
            else:
                name = 'not recognized'
                self.det.configure(text = '%s'%name)
        return 
    
    def video_Loop(self):
        #self.r = sr.Recognizer()
        frame, gr = self.camera()       
        fc = self.face_det(frame, gr)
        self.face_rec(frame, fc, gr)
        cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.inter(cv2img)
        #audio = self.aud()
        #self.recog(audio)
        
    def inter(self, cvimg):
        #Tkinter
        self.img = Image.fromarray(cvimg)
        imgtk = ImageTk.PhotoImage(image = self.img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image = imgtk)
        self.lmain.after(10, self.video_Loop)
        
    def aud(self):
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source)
            audio = self.r.listen(source)
        return audio
    
    def recog(self, audio):
        try:
            self.vc.configure(text = "You said " + self.r.recognize_google(audio))
            if self.r.recognize_google(audio) == 'hello':
                led.pow(1)
            if self.r.recognize_google(audio) == 'ok':
                led.pow(0)
            if self.r.recognize_google(audio) == 'quit':
                self.Quit()
            if self.r.recognize_google(audio) == 'right':
                led.pow(4)
            if self.r.recognize_google(audio) == 'left':
                led.pow(5)
        except sr.UnknownValueError:
            self.vc.configure(text = "Could not understand audio")
        except sr.RequestError as e:
            self.vc.configure(text = "Could not request results from Google Speech Recognition service; {0}".format(e))
    
    def ftrain(self):
        size = 1
        global im_width
        global im_height
        #take name
        fn_name = tkSimpleDialog.askstring('Name', 'Enter Name please')
        path = os.path.join(fn_dir, fn_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        
        # Generate name for image file
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
         if n[0]!='.' ]+[0])[-1] + 1

        #start taking pics
        tkMessageBox.showinfo('Message', 'We will take 20 samples to train')
        # The program loops until it has 20 images of the face.
        count = 0
        pause = 0
        count_max = 20
        while count < count_max:
            
            frame, gr = self.camera()
            # Get image size
            height, width, channels = frame.shape
            faces = self.face_det(frame, gr)
            # We only consider largest face
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * size for v in face_i]

                face = gr[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))

                # Remove false positives
                if(w * 6 < width or h * 6 < height):
                    print("Face too small")
                else:
                    # To create diversity, only save every fith detected image
                    if(pause == 0):
                        print("Saving training sample "+str(count+1)+"/"+str(count_max))
                    # Save image file
                        cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                        pin += 1
                        count += 1
                        pause = 1

            if(pause > 0):
                pause = (pause + 1) % 5
            
        tkMessageBox.showinfo('Message', 'Done!!')

        tkMessageBox.showinfo('Message', 'Training')
        
        (images, lables, names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(fn_dir, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1
        # Create a Numpy array from the two lists above
        (images, lables) = [numpy.array(lis) for lis in [images, lables]]


        # model = cv2.reateFisherFaceRecognizer()
        recognizer.train(images, lables)
        recognizer.save('/home/pi/Opencv/recognition/trainer.yml')
        tkMessageBox.showinfo('Message', 'Done!!')
        
    def Quit(self):
        self.cap.stop()
        self.root.quit()
        
pba = Application()
pba.root.mainloop()
