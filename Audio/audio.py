from gtts import gTTS
import os
# from playsound import playsound
text = "Welcome to my youtube channel: Rahul Singh Maures"

def return_speech(text):
    speech = gTTS(text=text,lang='en',slow=False)
    speech.save('new.mp3')

return_speech(text)

