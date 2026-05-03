# import kivy dependencies 

from kivy.app import App # base app class
from kivy.uix.boxlayout import BoxLayout  # vertical / horizontal box

# ui components
from kivy.uix.image import Image 
from kivy.uix.button import Button
from kivy.uix.label import Label

# import other kivy stuff
from kivy.clock import Clock # realtime feed from webcam
from kivy.graphics.texture import Texture # because we need our image to be 250px , 250px , 3 channels
from kivy.logger import Logger # to see how app performs 

# import other required dependencies : tf , cv 
import cv2
import tensorflow as tf
import layers as L1Dist # import layer.py
import os 
import numpy as np 


# Build the app layout 
class CamApp(App): #inheritence 

    def build(self):

        # define main layout components
        self.web_cam = Image(size_hint=(1,0.8)) # main input image
        self.button = Button(text='Verify', size_hint=(1,0.1))
        self.verification = Label(text = 'verification un-initiated', size_hint=(1,0.1))

        # add components and test the layout 
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        #setup video capture  device
        self.capture = cv2.VideoCapture(0)

        # real time feed
        Clock.schedule_interval(self.update, 1.0/33.0)


        return layout


    #run continuously to get webcam feed
    def update(self,*args):

        #read frame(numpy array)from opencv
        ret , frame = self.capture.read()
        frame = [120:120+250, 200:200+250]

        # flip horizontal and convert image to texture
        buf = cv2.flip(frame,0).tostring() # flip image horizlly -> cnvrt to string
        
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        
        # img -> convert it into texture -> rencer it inside app
        img_texture.blit_buffer(buf, colorfmt ='bgr', bufferfmt = 'ubyte')

        #converting raw opencv image -> array to a texture for rendering -> set image = that texture
        self.web_cam.texture = img_texture










if __name__ == '__main__':
    CamApp().run()

