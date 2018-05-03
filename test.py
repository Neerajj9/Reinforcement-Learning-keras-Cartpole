import gym
import numpy
from keras.layers import Dense
from keras.models import Sequential
import random
from collections import Counter
from statistics import mode,median,mean
from keras.layers import Dropout
from keras.models import model_from_json
import h5py

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 5000

score_req = 80
initial_game = 50000

load_json = open("weights.json",'r')
loaded = load_json.read()
load_json.close()
load = model_from_json(loaded)
load.load_weights("model.h5")
print("Loaded")
model = load

model.compile(loss = "categorical_crossentropy",optimizer = 'rmsprop' , metrics=['accuracy'])

scores = []
choices = []
for ep in range(100):
    score = 0
    game_mem = []
    env.reset()
    prev = []
    for _ in range(goal_steps):

        env.render()
        if len(prev)==0:
            action = random.randrange(0,2)

        else:
            action = numpy.argmax(model.predict(prev.reshape(-1,len(prev))))

        choices.append(action)

        ob,rew,done,info = env.step(action)

        game_mem.append([ob,action])

        prev = ob

        score = score + rew

        if done == True:
            print(score)
            break

    scores.append(score)


print("average : ",sum(scores)/len(scores))
print(score_req)
