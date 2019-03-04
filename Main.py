import gym
from betterCart import betterCart
import matplotlib.pyplot as plt
from scipy import stats
import time

n = 100000
tt = 200
yi = []
yf = []
r = 0
l = 0
env = gym.make('CartPole-v0')
learner = betterCart()
for i_episode in range(n):
	observation = env.reset()
	if 0:#i_episode % 100 == 0:
		for t in range(tt):
			env.render()
			print(observation)
			initialState = observation
			action = learner.nextAction(initialState)
			observation, reward, done, info = env.step(action)
			learner.updateQ(initialState, action, observation)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
	else:
		for t in range(tt):
			initialState = observation
			action = learner.nextAction(initialState)
			observation, reward, done, info = env.step(action)
			learner.updateQ(initialState, action, observation)
			if done:
				break

x = range(100)
for i_episode in x:
	observation = env.reset()
	t=0
	for t in range(tt):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			#print("Episode finished after {} timesteps".format(t+1))
			break
	yi.append(t)

x = range(100)
for i_episode in x:
	observation = env.reset()
	t=0
	for t in range(tt):
		env.render()
		action = learner.nextAction(initialState)
		if action == 0:
			l+=1
		else:
			r+=1
		observation, reward, done, info = env.step(action)
		if done:
			#print("Episode finished after {} timesteps".format(t+1))
			break
	yf.append(t)

env.close()

print('The average time started as: ', sum(yi)/100, " and finished as: ", sum(yf)/100)
# print(l , '       ',r)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# line = slope*x+intercept
# print("the slope is: ", slope)
# plt.plot(x,y,'o', x, line)
# plt.show()