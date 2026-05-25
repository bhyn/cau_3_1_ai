# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        # self.values[state]는 현재까지 계산한 V(s) 값입니다.
        # util.Counter는 없는 state를 물어봐도 기본값 0을 돌려줍니다.
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        for _ in range(self.iterations):
            # 이번 반복에서 새로 계산한 V_{k+1} 값을 여기에 따로 저장합니다.
            # 바로 self.values를 고치면 같은 반복 안에서 값이 섞이기 때문입니다.
            nextValues = util.Counter()

            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)
                if len(actions) == 0:
                    # terminal state처럼 가능한 행동이 없으면 미래 보상도 없으므로 V(s)=0입니다.
                    nextValues[state] = 0
                    continue

                # Bellman update:
                # 가능한 행동들 중 Q(s,a)가 가장 큰 값을 현재 state의 새 value로 사용합니다.
                nextValues[state] = max(
                    self.computeQValueFromValues(state, action)
                    for action in actions
                )

            # 한 번의 반복이 끝난 뒤에만 전체 value 테이블을 교체합니다.
            self.values = nextValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0

        # Q(s,a)는 가능한 nextState마다
        # 확률 * (즉시 보상 + 할인된 다음 state value)를 모두 더한 값입니다.
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            # 행동할 수 없는 terminal state에서는 정책도 없습니다.
            return None

        # 모든 legal action을 직접 확인합니다.
        # Counter.argMax만 쓰면 아직 Counter에 없는 행동을 놓칠 수 있습니다.
        bestAction = None
        bestValue = float('-inf')
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
