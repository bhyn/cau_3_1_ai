# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # Q(s,a)를 저장하는 테이블입니다.
        # key는 (state, action)이고, 처음 보는 쌍은 util.Counter 덕분에 0으로 시작합니다.
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # 직접 dict를 쓰지 않고 Counter를 쓰면 unseen action의 Q값이 자동으로 0입니다.
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            # terminal state에서는 고를 행동이 없으므로 max Q값을 0으로 봅니다.
            return 0.0

        # V(s) = max_a Q(s,a)
        # 여기서도 반드시 getQValue를 써야 ApproximateQAgent가 같은 로직을 재사용할 수 있습니다.
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            # terminal state에서는 실제로 할 행동이 없습니다.
            return None

        bestValue = self.computeValueFromQValues(state)
        # 같은 최고 Q값을 가진 행동이 여러 개라면 모두 후보로 모읍니다.
        bestActions = [
            action for action in legalActions
            if self.getQValue(state, action) == bestValue
        ]
        # 동점은 랜덤으로 깨야 한 방향으로만 치우치지 않습니다.
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # 현재 state에서 가능한 행동 목록을 먼저 가져옵니다.
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
            return None

        # epsilon 확률로는 탐험(explore): 아무 legal action이나 랜덤 선택합니다.
        # 나머지 확률로는 활용(exploit): 현재 Q값 기준 최선의 행동을 선택합니다.
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        oldQValue = self.getQValue(state, action)

        # sample은 "이번 경험으로 관측한 목표값"입니다.
        # 즉시 reward + discount * 다음 state에서 기대되는 최고 Q값입니다.
        sample = reward + self.discount * self.computeValueFromQValues(nextState)

        # Q-learning update:
        # 기존 Q값을 일부 남기고, 새 sample 쪽으로 alpha만큼 이동합니다.
        self.qValues[(state, action)] = (
            (1 - self.alpha) * oldQValue + self.alpha * sample
        )

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        # Approximate Q-learning에서는 Q(s,a)를 테이블에 직접 저장하지 않고
        # feature 값들과 weight들의 내적(dot product)으로 계산합니다.
        return self.weights * features

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)

        # correction은 현재 예측 Q값이 목표값과 얼마나 차이 나는지입니다.
        # 일반 Q-learning의 (sample - oldQValue)와 같은 역할을 합니다.
        correction = (
            reward
            + self.discount * self.computeValueFromQValues(nextState)
            - self.getQValue(state, action)
        )

        # 각 weight는 해당 feature가 얼마나 크게 나타났는지에 비례해서 조정됩니다.
        for feature, value in features.items():
            self.weights[feature] += self.alpha * correction * value

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
