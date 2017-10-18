import pygame
import os
import random
import torch
from pytools import Tools
from brain import AI_brain

import cv2
import pygame.camera

os.environ['SDL_VIDEO_CENTERED'] = '1'

# TO DO

# LSTM - RL joint training

class Filter(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Filter, self).__init__()
        self.relu = torch.nn.ReLU(True)

        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3,kernel_size= 3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3,kernel_size= 3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3,kernel_size= 3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=3, out_channels=3,kernel_size= 3, padding=1)
        self.relu = torch.nn.ReLU(True)
        self.init = True
        self.layer = None
        self.apply(self.initialize)
        self.init = False

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # maybe put this in the middle

        return x

    def initialize(self,m):
        #print(m)
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if (m==self.layer or self.init):
                k = 1/(27.0) #  (1/27.0, 0.025)
                s = 0.2
                m.weight.data.normal_(k,s)

            #print (m.weight.data)
        #convParams = list(self.conv.parameters())
        #convK, convB = convParams
        #print(convK.size(), convB.size())


class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class FilterCamera(torch.nn.Module):
    def __init__(self, c=6,r=3, kernels = 3):
        super(FilterCamera, self).__init__()
        self.relu = torch.nn.ReLU(True)

        self.convlist = []
        self.convlist.append(torch.nn.Conv2d(in_channels=3, out_channels=kernels,kernel_size= 3, padding=1))
        for i in range(c-2):
            self.convlist.append(torch.nn.Conv2d(in_channels=kernels, out_channels=kernels,kernel_size= 3, padding=1))

        self.convlist.append(torch.nn.Conv2d(in_channels=kernels, out_channels=3,kernel_size= 3, padding=1))
        self.conv = ListModule(*self.convlist)

        # Conv1d, then maxpooling conv1d, maxpooling etc

        self.c = c
        self.r = r

        self.relu = torch.nn.ReLU(True)
        self.init = True
        self.layer = None
        self.apply(self.initialize)
        self.init = False

    def forward(self, x):
        c = 0
        for i in range(self.r*self.c):
            x = self.convlist[c](x)
            c = (c + 1) % self.c
        # maybe put this in the middle

        return self.relu(x)

    def initialize(self,m):

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            k = 1/(27.0)
            s = 0.2
            m.weight.data.normal_(k,s)


class SceneBase:
    def __init__(self):
        self.next = self

    def Begin(self):
        print("uh-oh, you didn't override this in the child class")

    def ProcessEvents(self):
        print("uh-oh, you didn't override this in the child class")

    def Update(self):
        print("uh-oh, you didn't override this in the child class")

    def Render(self):
        print("uh-oh, you didn't override this in the child class")

    def Terminate(self):
        self.next = None


class Capture(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)


    def get(self):
        ret, frame = self.cap.read()
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return color


class AugmentedReality(SceneBase):

    def __init__(self, last_scene):
        super().__init__()
        self.last_scene = last_scene
        self.screen = None
        self.screen_size = None
        self.capture = Capture()
        self.next = None
        self.clock = 20
        self.filter = FilterCamera().cuda()

    def Begin(self):
        self.next = self
        self.screen_size = 640, 480
        self.screen = pygame.display.set_mode((self.screen_size[0], self.screen_size[1]), pygame.SRCALPHA)
        return self

    def ProcessEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.Terminate()
            if event.type == pygame.KEYDOWN:
                mykey = pygame.key.name(event.key)
                if mykey == "r":
                    self.filter.apply(self.filter.initialize)

    def Update(self):
        self.image = self.capture.get()
        self.image = self.image.swapaxes(0,2)
        self.tensor_image = torch.autograd.Variable(torch.from_numpy(self.image)).cuda()


        self.e = torch.autograd.Variable(torch.rand((self.screen_size[0], self.screen_size[1], 3))).cuda()
        self.e2 = torch.autograd.Variable(torch.rand((self.screen_size[0], self.screen_size[1], 3))).cuda()
        self.e = torch.add(self.e,-0.5)
        epsilon_distortion = 0 # 0.7
        self.e = torch.mul(self.e,torch.mul(self.e2,epsilon_distortion))

        self.e = self.e.byte()
        self.tensor_image = torch.add(self.tensor_image,self.e)

        self.tensor_image = self.tensor_image.float()
        self.tensor_image = self.tensor_image.unsqueeze(0)
        self.tensor_image = self.filter(self.tensor_image)
        self.tensor_image = torch.clamp(self.tensor_image,0,255)
        self.tensor_image = self.tensor_image.squeeze(0)
        self.tensor_image = self.tensor_image.data.cpu().numpy()
        self.tensor_image = self.tensor_image.swapaxes(0,2)
        self.tensor_image = self.tensor_image.swapaxes(0,1)


    def Render(self):
        self.screen.fill((0, 0, 0))
        pygame.surfarray.blit_array(self.screen,self.tensor_image)


    def Terminate(self):
        self.next = None

class PhysicsSimulator(SceneBase):

    def __init__(self, last_scene):
        super().__init__()
        self.last_scene = last_scene
        self.screen = None
        self.screen_size = None
        self.field_matrix = None
        self.e = None
        self.e2 = None
        self.s = None
        self.clock = 60
        self.nonrandom = False
        self.filter = Filter().cuda()

    def Begin(self):

        self.screen_size = 1280, 768
        self.screen = pygame.display.set_mode((self.screen_size[0], self.screen_size[1]), pygame.SRCALPHA)
        self.screen.fill((0, 0, 0))
        self.next = self

        self.s = 96, 56
        self.field_matrix = torch.autograd.Variable(torch.abs(torch.rand((self.s[0],self.s[1],3))), volatile=True,requires_grad=False).cuda()

        return self

    def Render(self):
        self.screen.fill((0, 0, 0))

        t0 = self.field_matrix[:,:].data.cpu()

        for i in range(self.s[0]):
            pygame.draw.line(self.screen,(20,20,20),(2+i*20,0),(2+i*20,self.s[1]*20))
        for j in range(self.s[1]):
            pygame.draw.line(self.screen,(20,20,20),(0,2+j*20),(self.s[0]*20,2+j*20))

        for i in range(self.s[0]):
            for j in range(self.s[1]):
                motion = 5
                pygame.draw.circle(self.screen, (255*t0[i][j][0],255*t0[i][j][1],128*t0[i][j][2]), (2+i*20+int(motion*t0[i][j][0]-motion/2), 2+j*20+int(motion*t0[i][j][1]-motion/2)), int(7*t0[i][j][2])+1, 1)

    def Update(self):

        self.e = torch.autograd.Variable(torch.rand((self.s[0], self.s[1], 3))).cuda()
        self.e2 = torch.autograd.Variable(torch.rand((self.s[0], self.s[1], 3))).cuda()
        self.e = torch.add(self.e,-0.5)
        epsilon_distortion = 0.1 # 0.7
        self.e = torch.mul(self.e,torch.mul(self.e2,epsilon_distortion))

        self.field_matrix = torch.add(self.field_matrix,self.e)

        self.field_matrix = self.field_matrix.permute(2,0,1)
        self.field_matrix = self.field_matrix.unsqueeze(0)
        if self.nonrandom:
            self.field_matrix = self.filter(self.field_matrix)
        self.field_matrix = self.field_matrix.squeeze(0)
        self.field_matrix = self.field_matrix.permute(1,2,0)

        #print(self.field_matrix.size())

        self.field_matrix = torch.clamp(self.field_matrix,0,1)

    def ProcessEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.Terminate()

            if event.type == pygame.KEYDOWN:
                mykey = pygame.key.name(event.key)
                if mykey == "space":
                    self.next = self.last_scene
                if mykey == "1":
                    self.filter.layer = self.filter.conv
                    self.filter.apply(self.filter.initialize)
                if mykey == "2":
                    self.filter.layer = self.filter.conv2
                    self.filter.apply(self.filter.initialize)
                if mykey == "3":
                    self.filter.layer = self.filter.conv3
                    self.filter.apply(self.filter.initialize)
                if mykey == "4":
                    self.filter.layer = self.filter.conv4
                    self.filter.apply(self.filter.initialize)
                if mykey == "f":
                    pygame.display.set_mode((1920,1080),pygame.FULLSCREEN)
                    print("toggle fullscreen")
                if mykey == "e":
                    pygame.display.set_mode((self.screen_size[0],self.screen_size[1]))
                    print("toggle fullscreen")
                if mykey == "r":
                    self.nonrandom = True


class CyberTerm(SceneBase):

    def __init__(self):
        super().__init__()
        self.screen_size = None
        self.screen = None
        self.next = self
        self.alpha_accelerate = None
        self.alpha = None
        self.Tools = None
        self.AI_brain = None
        self.current_im = None
        self.clock = 1

    def Begin(self):

        self.screen_size = 1280, 768
        self.screen = pygame.display.set_mode((self.screen_size[0], self.screen_size[1]), pygame.SRCALPHA)
        self.screen.fill((0, 0, 0))
        self.next = self
        self.alpha_accelerate = 1
        self.alpha = 0
        self.Tools = Tools()
        self.AI_brain = AI_brain(self.Tools)

        self.current_im = self.Tools.get_random_image()
        if not pygame.mixer.music.get_busy():
            self.Tools.set_music(style="cyberpunk")
        text = "Welcome dear!"
        self.AI_brain.time_to_wait = len(text) * 6
        self.Tools.set_AI_text("AI Assistant: " + text)
        self.AI_brain.time_to_write = 0

        return self

    def AI_text(self, text):
        if "excellent" in text:
            self.AI_brain.add_mind_state("Happy to serve")
        elif "horrible" in text:
            self.AI_brain.add_mind_state("Self-deprecation")
        elif "physics simulation" in text:
            self.next = scenes[1]
        elif "command" in text:
            self.AI_brain.AI_hard_command(text)

    def process_text(self, text):
        text = text.lower()
        text = ''.join(x for x in text if x not in '.')
        text = ''.join(x for x in text if x in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        self.AI_text(text)


    def ProcessEvents(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.Terminate()
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


    def Render(self):

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

    def Update(self):
        self.AI_brain.process_mind_state()

CT = CyberTerm()
PS = PhysicsSimulator(CT)
AR = AugmentedReality(CT)
scenes = (CT, CT, CT)

def Run():

    pygame.init()
    pygame.mixer.init()

    screen_size = 1280, 768
    pygame.display.set_mode((screen_size[0], screen_size[1]), pygame.SRCALPHA)

    clock = pygame.time.Clock()
    active_scene = scenes[2].Begin()

    while active_scene != None:

        active_scene.ProcessEvents()
        active_scene.Update()
        active_scene.Render()
        pygame.display.flip()
        if active_scene.next is None:
            exit()

        if active_scene != active_scene.next:
            active_scene.next.Begin()
        active_scene = active_scene.next

        clock.tick(active_scene.clock)

Run()
