import streamlit as st
import tensorflow as tf
import tempfile
import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2
SEQUENCE_LENGTH = 20
def frames_extraction(video_reader):
    frames_list = []
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/20), 1)
    for frame_counter in range(20):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read() 
        if not success:
            break
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list
model_path="C:/Users/SRINIVAS/Downloads/capstone/LRCN_model___Date_Time_2022_09_28__15_17_39___Loss_1.3025104999542236___Accuracy_0.7094762921333313.h5"
st.title("HUMAN ACTIVITY RECOGNITION USING TENSORFLOW (CNN+LSTM)")
st.markdown("""
<style>
body {
  background: #ff0099; 
  background: -webkit-linear-gradient(to right, #ff0099, #493240); 
  background: linear-gradient(to right, #ff0099, #493240); 
}
</style>
    """, unsafe_allow_html=True)
st.title("TRAINED ACTIONS----")
CLASSES_LIST = ["BaseballPitch","Basketball","BenchPress","Biking","Billiards","BreastStroke","CleanAndJerk","Diving","Drumming","Fencing","GolfSwing","HighJump","HorseRace","HorseRiding","HulaHoop","JavelinThrow","JugglingBalls","JumpingJack","JumpRope","Kayaking","Lunges","MilitaryParade","Mixing","Nunchucks","PizzaTossing", "PlayingGuitar", "PlayingPiano", "PlayingTabla","PlayingViolin","PoleVault","PommelHorse","PullUps","RockClimbingIndoor","RopeClimbing","Rowing","SalsaSpin","SkateBoarding","Skiing","Skijet","SoccerJuggling","Swing","TaiChi","TennisSwing","ThrowDiscus","TrampolineJumping","VolleyballSpiking","WalkingWithDog","YoYo"]
st.table(CLASSES_LIST)
upload = st.file_uploader('Upload a video to predict action ;)')
if upload is not None:
  st.success('Video Uploaded Successfully..')
  tfile = tempfile.NamedTemporaryFile(delete=False) 
  tfile.write(upload.read())
  vf = cv2.VideoCapture(tfile.name)
  frames = frames_extraction(vf)
  f=[]
  
  f.append(frames)
  f = np.asarray(f)
  model=tf.keras.models.load_model(model_path)
  st.title(CLASSES_LIST[np.argmax(model.predict(f))])


      