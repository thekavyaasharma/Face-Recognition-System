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
class CamApp(App):

    def build(self):
        pass

if __name__ == '__main__':
    CamApp.run()

