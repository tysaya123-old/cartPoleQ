import math
import numpy as np
import random

class qLearnerCart():
	qs = 0;
	ns = 0;
	xs, vs, ts, ws, actions = 51, 41, 27, 61, 2

	def __init__(self, gamma):
		#, xs, vs, ts, ws , actions, 
		#x from [-2.4,2.4] by .1 so [0, 48]
		#v from [-2,2] by .1 with -2 including those smaller and 2 including those larger so [0, 40]
		#t(heta) from [-12,12] by 1 so [0, 24]
		#w(angular velocity) from [-3,3] by .1 with -3 including those smaller and 3 including those larger so [0, 60]

		self.qs = np.ndarray(shape = (self.xs, self.vs, self.ts, self.ws, self.actions), dtype = float)
		self.ns = np.ndarray(shape = (self.xs, self.vs, self.ts, self.ws, self.actions), dtype = int)
		self.g = gamma

	def chooseBest(self, state):
		x, v, t, w = state
		x, v, t, w = self.convertValues(x, v, t, w)

		q0 = self.qs[x][v][t][w][0]
		q1 = self.qs[x][v][t][w][1]
		if q0 > q1:
			return 0
		else:
			return 1

	def chooseNextAction(self, state):
		x, v, t, w = state
		x, v, t, w = self.convertValues(x, v, t, w)
		rand = random.random()
		p = (self.qs[x][v][t][w][0] + 1)/(np.sum(self.qs[x][v][t][w]) + 2)
		if p < rand:
			return 0
		else:
			return 1

	def updateQ(self, state, a, state2, reward):
		x, v, t, w = state
		x2, v2, t2, w2 = state2
		x, v, t, w = self.convertValues(x, v, t, w)
		x2, v2, t2, w2 = self.convertValues(x2, v2, t2, w2)

		reward = abs(13 - t) - abs(13 - t2)

		alpha = 1/(1 + self.ns[x][v][t][w][a])
		q = self.qs[x][v][t][w][a]
		qp = max(self.qs[x2][v2][t2][w2])
		self.qs[x][v][t][w][a] = (1 - alpha)*q + alpha*(reward + self.g*qp)


	def convertValues(self, x, v, t, w):
		if x < -2.4:
			x = -2.5
		if x > 2.4:
			x = 2.5
		x = int(round(x*10) + (self.xs-1)/2)
		if v <= -2:
			v = -2
		if v >= 2:
			v = 2
		v = int(round(v*10) + (self.vs-1)/2)
		t = math.degrees(t)
		if t < -12:
			t = -13
		if t > 12:
			t = 13
		t = int(round(t) + (self.ts-1)/2)
		if w <= -3:
			w = -3
		if w >= 3:
			w = 3
		w = int(round(w*10) + (self.ws-1)/2)
		return x, v, t, w
