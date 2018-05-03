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
goal_steps = 8000

score_req = 80
initial_game = 50000


"""
for ep in range(5):
    env.reset()
    #print("H")
    for _ in range(200):
        env.render()
        action = env.action_space.sample()

        ob,rew,done,info = env.step(action)
        print(ob)
        if done:
            break
"""

def generate_data():
    training_data = []
    scores = []
    acc_scores = []

    for _ in range(initial_game):

        score = 0
        game_mem = []
        prev_obv = []

        for _ in range(goal_steps):
            action = random.randrange(0,2)
            ob, rew, done, info = env.step(action)

            if len(prev_obv)>0:
                game_mem.append([prev_obv,action])

            prev_obv = ob

            score = score + rew
            if done:
                break


        if score >= score_req:
            acc_scores.append(score)

            for data in game_mem:
                if data[1]==1:
                    out = [0,1]
                elif data[1]==0:
                    out=[1,0]

                training_data.append([data[0],out])


        env.reset()

        scores.append(score)

    X = numpy.array(training_data)
    numpy.save('data.npy',X)

    print('Average accepted score:', mean(acc_scores))
    print('Median score for accepted scores:', median(acc_scores))
    print(Counter(acc_scores))

    return X




def neural_net(INP):

    X = numpy.array([i[0] for i in INP])
    Y = numpy.array([i[1] for i in INP])

    model = Sequential()

    model.add(Dense(128, input_dim= 4 , activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X,Y,epochs=10,batch_size=100)
    return model

def save_weights(model):

    model_save = model.to_json()
    with open("weights.json","w") as json_file:
        json_file.write(model_save)

    model.save_weights("model.h5")
    print("SAVED")

INP = generate_data()
model = neural_net(INP)
save_weights(model)

scores = []
choices = []
for ep in range(15):
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

