import random
import glob
import pygame
import os

import _thread
import win32com.client as wincl

class Tools():

    def __init__(self):
        self.mlp = "./resources/background music/" # music library path

        self.playlist = glob.glob("./resources/background music/*.ogg")

        # all lower case since we convert from terminal
        self.music_names = ["character creation", "prelude"]
        self.music_composers = ["rachmaninoff","chopin","bach","unknown"]
        self.music_styles = ["cyberpunk", "classical","jazz"]

        self.ilp  ='./resources/background art/'  # image library path
        self.image_library = {}
        self.font = pygame.font.SysFont("Terminal", 24)
        self.line_height = 15
        self.line_length = 120 # Never use a word larger than this, it can go over the line if it's in the end
        self.text = ""
        self.texts = []
        self.uppercase = False

        self.AItext = ""
        self.AItexts = []
        self.speak = wincl.Dispatch("SAPI.SpVoice")
        self.text_brightness = 128
        self.narration = self.font.render(' '.join("Narration: "), True, (self.text_brightness, self.text_brightness, self.text_brightness))

        self.volume = 1
        self.set_volume(self.volume)

    def set_volume(self, volume):

        try:
            assert(0 <= volume <=10)
        except AssertionError:
            print("Exception: Volume is not within proper range [0,1], muting sounds")
            volume = 0.0

        self.volume = volume
        pygame.mixer.music.set_volume(self.volume/10.0)

    def set_music(self, style=None, composer=None, name=None):
        songs = self.playlist[:]
        songs = [x.lower() for x in songs]

        mysongs = []

        for song in songs:
            if not name is None and name in song.replace("_"," "):
                mysongs.append(song)
                break
            elif not composer is None and composer in song.replace("_"," "):
                mysongs.append(song)
                break
            elif not style is None and style in song.replace("_"," "):
                mysongs.append(song)
                break

        if len(mysongs) > 0:
            mysong = random.choice(mysongs)
            pygame.mixer.music.load(mysong)
            pygame.mixer.music.play(loops=-1,start=0)

    def get_random_image(self):
        images = glob.glob(self.ilp+"/*.jpg")
        images = [x for x in images if "basement" in x]
        name = random.choice(images)
        image = self.image_library.get(name)

        if image is None:
            canonicalized_path = name.replace('/', os.sep).replace('\\', os.sep)
            image = pygame.image.load(canonicalized_path)
            self.image_library[name] = image
        return image

    def get_image(self, name):
            name = self.ilp+name
            image = self.image_library.get(name)

            if image is None:
                    canonicalized_path = name.replace('/', os.sep).replace('\\', os.sep)
                    image = pygame.image.load(canonicalized_path)
                    self.image_library[name] = image
            return image

    def speak_independently(self, threadName, text, featureEnabled = False):
        if featureEnabled:
            self.speak.Speak(text)

    def set_AI_text(self, text):
        if text == self.AItext:
            return self.AItext
        else:
            self.AItext = text
            words = text.split(" ")

            self.AItexts = []
            mylength = 0
            line = []
            i = 0
            for word in words:
                mylength += len(word)
                if mylength < self.line_length:
                    line.append(word)
                else:
                    mylength = 0
                    line.append(word)
                    self.AItexts.append((self.font.render(' '.join(line), True, (self.text_brightness, self.text_brightness, self.text_brightness)), i))
                    i += 1
                    line = []

            self.AItexts.append((self.font.render(' '.join(line), True, (self.text_brightness, self.text_brightness, self.text_brightness)), i))

            _thread.start_new_thread(self.speak_independently, ( "Myname", text.strip("AI Assistant: ") ) )
            # Separate thread would be better, since this blocks

            return self.AItexts


    def set_text(self, text):

        # Break text into lines based on length. If text is equal to previous text, do not change the renderer
        if text == self.text:
            return self.texts
        else:
            self.text = text
            words = text.split(" ")

            self.texts = []
            mylength = 0
            line = []
            i = 0
            for word in words:
                mylength += len(word)
                if mylength < self.line_length:
                    line.append(word)
                else:
                    mylength = 0
                    line.append(word)
                    self.texts.append((self.font.render(' '.join(line), True, (self.text_brightness, self.text_brightness, self.text_brightness)), i))
                    i += 1
                    line = []

            self.texts.append((self.font.render(' '.join(line), True, (self.text_brightness, self.text_brightness, self.text_brightness)), i))
            return self.texts

    def add_text(self, key):
        text = self.text
        text += key
        self.set_text(text)

    def delete_text(self,i):

        original_text = str(self.text)
        text = self.text[:-i]
        self.set_text(text)
        return original_text

