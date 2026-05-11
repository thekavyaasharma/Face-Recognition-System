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
from layers import L1Dist # import layer.py
import os 
import numpy as np 


# Build the app layout 
class CamApp(App): #inheritence 

    def build(self):

        # define main layout components
        self.web_cam = Image(size_hint=(1,0.8)) # main input image
        self.button = Button(text='Verify',on_press=self.verify, size_hint=(1,0.1))
        self.verification_label = Label(text = 'Click the verify button to mark attendance.', size_hint=(1,0.1))

        # add components and test the layout 
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #Load keras siamese model 
        self.model = tf.keras.models.load_model('siameseModelv2.h5', custom_objects={'L1Dist':L1Dist})

        #setup video capture  device
        self.capture = cv2.VideoCapture(0)

        # real time feed
        Clock.schedule_interval(self.update, 1.0/33.0)


        return layout


    #run continuously to get webcam feed
    def update(self,*args):

        #read frame(numpy array)from opencv
        ret , frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250,:]

        # flip horizontal and convert image to texture
        buf = cv2.flip(frame,0).tobytes() # flip image horizlly 
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        
        # img -> convert it into texture -> rencer it inside app
        img_texture.blit_buffer(buf, colorfmt ='bgr', bufferfmt = 'ubyte')

        #converting raw opencv image -> array to a texture for rendering -> setting image equal that texture
        self.web_cam.texture = img_texture


    #before passing img to model preprocess it Bring preprocess fxn from FCR.ipynb

    # Load image from file then convert it into 100px, 100px
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)  # read img using file path treeated using bytes like obj 
        img = tf.io.decode_jpeg(byte_img)          # decode jpeg -> load img 
        
        img = tf.image.resize(img,(100,100))    # resize img -> preprocessing (100px, 100px, 3 channels) as per paper 
        img = img / 255.0              # divide it by 255 so it performs scaling and it returnns the image 0-1
        return img


    # Bring verifrication fxn from FCR.ipynb to verify person 
    # 4 args :  siameseNN , metric above which pred is considered +ve class, ratio of +ve pred over total +ve samples(30/50)
    
    def verify(self, *args):

        # specify thresholds 
        detection_threshold = 0.99
        verification_threshold = 0.8

        # capture img from webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # build results array: store results in array
        results = []

        # loop through every 50 images in verif folder (full cycle loop)
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img = self.preprocess(os.path.join('application_data','input_image', 'input_image.jpg')) # loads image -> resive -> convert pcx to no
            validation_img = self.preprocess(os.path.join('application_data','verification_images', image)) # same for valid image

            # make pred 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        detection =  np.sum(np.array(results)>detection_threshold)
        
        verification = detection / len(os.listdir(os.path.join('application_data','verification_images')))
        verified = verification > verification_threshold 

        # set verification text 
        self.verification_label.text = "Successfully Verified!" if verified == True else "Failed to Verify! Try Again."

        # Log out details 
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.3))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))


        return results , verified
    
    






if __name__ == '__main__':
    CamApp().run()

