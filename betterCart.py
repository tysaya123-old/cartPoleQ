import numpy as np
import random
class betterCart:
	qs = 0 #holds all Q values
	ns = 0 #holds the number of times a state-action pair has been visited
	max_x, max_v, max_t, max_w = 50, 40, 40, 62 #The max index for an obsevation
	actions = 2 #number of possible action
	g = .9 #gamma value
	k = 1 #const used in nextAction()

	def __init__(self):
		self.qs = np.ndarray(shape = (self.max_x, self.max_v, self.max_t, self.max_w, self.actions), dtype = float)
		self.ns = np.ndarray(shape = (self.max_x, self.max_v, self.max_t, self.max_w, self.actions), dtype = int)

	def updateQ(self, stateI, action, stateF):
		xi, vi, ti, wi = self.convertToIndexs(stateI)
		xf, vf, tf, wf = self.convertToIndexs(stateF)

		reward = abs(tf) - abs(ti)

		self.qs[xi][vi][ti][wi][action] = reward + self.g*max(self.qs[xf][vf][tf][wf])
		# alpha = 1/(1 + self.ns[xi][vi][ti][wi][action])
		# qi = self.qs[xi][vi][ti][wi][action]
		# qf = max(self.qs[xf][vf][tf][wf])
		# self.qs[xi][vi][ti][wi][action] = (1 - alpha)*qi + alpha*( reward + self.g*qf )

	#Use k^qi / sum(k^qj) to probabalistically choose the next action
	def nextAction(self, state):
		x, v, t, w = self.convertToIndexs(state)

		r = random.random()

		q0 = self.qs[x][v][t][w][0]
		q1 = self.qs[x][v][t][w][1]
		#probability of choosing 0
		p0 = self.k**q0/(self.k**q0 + self.k**q1)

		if r < p0:
			return 0
		else:
			return 1

	#Converts state values into whole numbers which correspond to their index in the qs and ns
	def convertToIndexs(self, state):
		x, v, t, w = state
		if x < -2.4:
			x = -2.5
		if x > 2.4:
			x = 2.5
		x = int(round(x*10) + (self.max_x)/2) - 1
		if v <= -2:
			v = -2
		if v >= 2:
			v = 2
		v = int(round(v*10) + (self.max_v)/2) - 1
		if t < -.2:
			t = -.2
		if t > .2:
			t = .2
		t = int(round(t*100) + (self.max_t)/2) - 1
		if w <= -3:
			w = -3
		if w >= 3:
			w = 3
		w = int(round(w*10) + (self.max_w)/2) - 1
		return x, v, t, w