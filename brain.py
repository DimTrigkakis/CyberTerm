import random

class AI_brain():

    def __init__(self,Tools):

        self.mind_state = []
        self.mind_reading = []

        self.Tools = Tools

        self.time_to_write = 0
        self.time_to_wait = 0

        self.pms = ["Self-deprecation","Happy to serve","done","impossible","confused command"] # possible mind states for RL and hardcoded words
        self.possible_action_templates = ["Write Response","Load Music","FB Message","Write Document", "Ask Google","Download Art","Remember"]
        # Possible actions implemented in the machine, the model must supply reasonable arguments

    # The mind states should actually be vectors
    # The actions to take should be done with RL
    # The changes of mindstates should be done with LSTM

    # Pipeline -> Process input with LSTM. Figure out how to change own LSTM of thoughts. Then use RL from LSTM of thoughts to output, based on thought importance.


    # Hard commands only

    def AI_hard_command(self,commandment):
        self.process_command(commandment)

    def process_command(self,commandment):
        bag_of_words = {"change music":["change","music"],"a little more sound":["raise","volume","a little"],"a little less sound":["lower","volume","a little"],"exit":["exit"],"way less sound":["lower","volume","a lot"],"less sound":["lower","volume"],"way more sound":["raise","volume","a lot"],"more sound":["raise","volume"],}
        # ordering matters in prefix code

        for key in bag_of_words.keys():
            valid = True
            for word in bag_of_words[key]:
                if word not in commandment:
                    valid = False

            if valid:
                if key == "change music":

                    for n in self.Tools.music_names:
                        if n in commandment:
                            self.Tools.set_music(name=n)
                            self.add_mind_state("done")
                            return
                    for n in self.Tools.music_composers:
                        if n in commandment:
                            self.Tools.set_music(composer=n)
                            self.add_mind_state("done")
                            return
                    for n in self.Tools.music_styles:
                        if n in commandment:
                            self.Tools.set_music(style=n)
                            self.add_mind_state("done")
                            return
                    self.add_mind_state("impossible")
                    return

                if key == "a little less sound":
                    if self.Tools.volume >= 1:
                        self.Tools.set_volume(self.Tools.volume-1)
                        self.add_mind_state("done")
                        return
                    else:
                        self.add_mind_state("impossible")
                        return

                if key == "a little more sound":
                    if self.Tools.volume <= 9:
                        self.Tools.set_volume(self.Tools.volume+1)
                        self.add_mind_state("done")
                        return
                    else:
                        self.add_mind_state("impossible")
                        return
                if key == "less sound":
                    if self.Tools.volume >= 2:
                        self.Tools.set_volume(self.Tools.volume-2)
                        self.add_mind_state("done")
                        return
                    else:
                        self.add_mind_state("impossible")
                        return

                if key == "more sound":
                    if self.Tools.volume <= 8:
                        self.Tools.set_volume(self.Tools.volume+2)
                        self.add_mind_state("done")
                        return
                    else:
                        self.add_mind_state("impossible")
                        return
                if key == "way less sound":
                    if self.Tools.volume >= 4:
                        self.Tools.set_volume(self.Tools.volume-4)
                        self.add_mind_state("done")
                        return
                    else:
                        self.add_mind_state("impossible")
                        return

                if key == "way more sound":
                    if self.Tools.volume <= 6:
                        self.Tools.set_volume(self.Tools.volume+4)
                        self.add_mind_state("done")
                        return
                    else:
                        self.add_mind_state("impossible")
                        return

                if key == "exit":
                    exit()
                    return

        self.add_mind_state("confused command")

    # Action templates

    def try_to_write(self,text,removals,pms): # all actions have this template
        if self.time_to_write > self.time_to_wait:
            self.time_to_wait = len(text) * 6
            self.Tools.set_AI_text("AI Assistant: " + text)
            self.time_to_write = 0
            removals.append(pms)

    # core functionality

    def execute_mind_state(self, this_state, removals):

        for pms in self.pms:
            if pms in this_state:

                if pms == self.pms[0]:
                    possible_responses = ["I am dissatisfied as well...","That was really bad...",
                                          "I learned something, at least..."]
                elif pms == self.pms[1]:
                    possible_responses = ["I am happy to have succeeded!","I knew I would make it!",
                                          "I learned something new and exciting!"]
                elif pms == self.pms[2]:
                    possible_responses = ["Ok, done!","I did the deed!",
                                          "Are you satisfied with that? Hopefully you are!"]
                elif pms == self.pms[3]:
                    possible_responses = ["That is an invalid request!","You cannot keep asking these things!",
                                          "I don't get you sometimes, that's just impossible..."]
                elif pms == self.pms[4]:
                    possible_responses = ["I didn't get that!?","I am confused, or maybe you are...",
                                          "What do you mean again?"]
                if len(possible_responses) > 0:
                    self.try_to_write(random.choice(possible_responses),removals,pms)

    def process_mind_state(self):

        removals = []

        self.time_to_write += 1
        if len(self.mind_state) == 0:
            if self.time_to_write > self.time_to_wait:
                text = "..."
                self.time_to_wait = len(text) * 6
                self.Tools.set_AI_text("AI Assistant: " + text)
                self.time_to_write = 0

        for this_state in self.mind_state:
            self.execute_mind_state(this_state, removals)

        self.mind_state = [x for x in self.mind_state if x not in removals]

    def add_mind_state(self, text):
        self.mind_state.append(text)
