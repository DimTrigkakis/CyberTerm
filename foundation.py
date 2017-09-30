import pygame
import os
import random
import torch
from pytools import Tools
from brain import AI_brain

os.environ['SDL_VIDEO_CENTERED'] = '1'

# TO DO

# LSTM - RL joint training

class CyberTerm():
    def __init__(self):

        pygame.init()
        pygame.mixer.init()
        self.alpha_accelerate = 1
        self.alpha = 0
        self.Tools = Tools()
        self.AI_brain = AI_brain(self.Tools)

        text = "Welcome dear!"
        self.AI_brain.time_to_wait = len(text) * 6
        self.Tools.set_AI_text("AI Assistant: " + text)
        self.AI_brain.time_to_write = 0

        self.current_im = self.Tools.get_random_image()
        self.screen_size = 1280, 768
        self.screen = pygame.display.set_mode((self.screen_size[0], self.screen_size[1]), pygame.SRCALPHA)
        self.quit = False
        self.Tools.set_music(style="cyberpunk")
        self.clock = pygame.time.Clock()

    def AI_text(self, text):
        if "excellent" in text:
            self.AI_brain.add_mind_state("Happy to serve")
        elif "horrible" in text:
            self.AI_brain.add_mind_state("Self-deprecation")
        elif "command" in text:
            self.AI_brain.AI_hard_command(text)

    def process_text(self, text):
        text = text.lower()
        text = ''.join(x for x in text if x not in '.')
        text = ''.join(x for x in text if x in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        self.AI_text(text)


    def event_triggers(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            if event.type == pygame.KEYDOWN:
                special_cases = ["space","right shift","backspace","return"]
                mykey = pygame.key.name(event.key)
                if mykey not in special_cases:
                    if self.Tools.uppercase:
                        mykey = mykey.upper()
                    self.Tools.add_text(mykey)
                elif mykey == "space":
                    self.Tools.add_text(" ")
                elif mykey == "right shift":
                    self.Tools.uppercase = True
                elif mykey == "backspace":
                    self.Tools.delete_text(1)
                elif mykey == "return":
                    original_text = self.Tools.delete_text(len(self.Tools.text))
                    self.process_text(original_text)
            if event.type == pygame.KEYUP:
                mykey = pygame.key.name(event.key)
                if mykey == "right shift":
                    self.Tools.uppercase = False


    def draw_event(self):

        self.screen.fill((0, 0, 0))

        self.alpha += self.alpha_accelerate
        if self.alpha < 0 :
            self.alpha = 0

            self.current_im = self.Tools.get_random_image()
            self.alpha_accelerate *= -1

        if self.alpha > 255 :
            self.alpha = 255
            self.alpha_accelerate *= 0

        if random.random() > 0.999 and self.alpha == 255:
            self.alpha_accelerate = -5
        self.current_im = pygame.transform.scale(self.current_im, (self.screen_size[0]-200, self.screen_size[1]-250))
        current_imsize = self.current_im.get_rect().size

        # draw current image

        self.current_im.set_alpha(self.alpha)
        self.screen.blit(self.current_im, ((self.screen_size[0]-current_imsize[0])/2, 40))

        # draw current text
        self.screen.blit(self.Tools.narration,(( 100,30+ self.screen_size[1] - 200 + self.Tools.line_height * 0)))
        for text in self.Tools.texts:
            self.screen.blit(text[0],(( 125,35+ self.screen_size[1] - 200 + self.Tools.line_height * (text[1]+1))))

        for text in self.Tools.AItexts:
            self.screen.blit(text[0],(( 100, self.screen_size[1] - 200 + self.Tools.line_height * text[1])))

        pygame.display.flip()

    def update_event(self):

        self.AI_brain.process_mind_state()
        self.clock.tick(60)

    def main_loop(self):
        while not self.quit:
            self.event_triggers()
            self.draw_event()
            self.update_event()

CT = CyberTerm()
CT.main_loop()
