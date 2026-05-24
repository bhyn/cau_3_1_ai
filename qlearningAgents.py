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

def _nearestPoint(pos):
    """
    유령 위치가 가끔 0.5 단위로 있을 수 있어서, feature 계산용으로
    가장 가까운 격자 칸 좌표로 바꿔 줍니다.
    """
    return (int(pos[0] + 0.5), int(pos[1] + 0.5))

def _closestMazeDistance(start, targets, walls):
    """
    벽을 고려해서 start에서 targets 중 가장 가까운 곳까지의 거리를 구합니다.
    단순 Manhattan distance보다 실제 Pacman 길찾기에 더 잘 맞습니다.
    """
    targets = set(targets)
    if len(targets) == 0:
        return None

    start = _nearestPoint(start)
    if start in targets:
        return 0

    fringe = [(start[0], start[1], 0)]
    visited = set()
    while len(fringe) > 0:
        x, y, dist = fringe.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))

        for nextX, nextY in Actions.getLegalNeighbors((x, y), walls):
            if (nextX, nextY) in targets:
                return dist + 1
            fringe.append((nextX, nextY, dist + 1))

    return None

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

    # Q(s,a) = Q(s,a) + alpha * { reward + discount * max_a' Q(s',a') - Q(s,a) }
    # Q(s,a) = Q(s,a) + alpha * { reward + discount * V(s')           - Q(s,a) }

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
        # 어떤 행동을 할지를 고르는 부분이다. 아까 위의 computeValueFromQValues에서 최대가 되는 a에서의 V(s)를 구햇다면 여기서는 바로 그 a만을 구하는 거다. 
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

        # epsilon 확률로는 explore: 아무 legal action이나 랜덤 선택합니다.
        # 나머지 확률로는 exploit: 현재 Q값 기준 최선의 행동을 선택합니다.
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


class CompetitionExtractor(FeatureExtractor):
    """
    경쟁용 feature extractor입니다.

    SimpleExtractor보다 더 많은 정보를 봅니다:
      - 음식을 바로 먹는지
      - 가장 가까운 음식까지의 실제 거리
      - 남은 음식 수
      - 캡슐을 먹는지 / 캡슐까지 거리
      - 위험한 유령과의 거리
      - 먹을 수 있는 유령(scared ghost)과의 거리
      - Stop이나 바로 되돌아가기 같은 나쁜 움직임
    """

    def getFeatures(self, state, action):
        features = util.Counter()

        walls = state.getWalls()
        food = state.getFood()
        capsules = state.getCapsules()
        ghostStates = state.getGhostStates()
        width, height = walls.width, walls.height
        boardSize = float(width * height)

        successor = state.generatePacmanSuccessor(action)
        nextPos = successor.getPacmanPosition()
        nextFood = successor.getFood()
        nextCapsules = successor.getCapsules()

        features["bias"] = 1.0

        # 점수 변화는 음식 +10, 시간 -1, 죽음 -500 같은 즉시 결과를 담고 있습니다.
        features["score-change"] = (successor.getScore() - state.getScore()) / 100.0

        if successor.isWin():
            features["wins"] = 1.0
        if successor.isLose():
            features["loses"] = 1.0

        if action == Directions.STOP:
            features["stops"] = 1.0

        currentDirection = state.getPacmanState().getDirection()
        if action == Actions.reverseDirection(currentDirection):
            features["reverses"] = 1.0

        # 방금 움직여서 음식을 먹었는지 확인합니다.
        if successor.getNumFood() < state.getNumFood():
            features["eats-food"] = 1.0

        # 음식이 많이 남을수록 나쁘게 보도록 정규화해서 넣습니다.
        features["food-left"] = successor.getNumFood() / boardSize

        foodList = nextFood.asList()
        closestFoodDist = _closestMazeDistance(nextPos, foodList, walls)
        if closestFoodDist is not None:
            features["closest-food"] = closestFoodDist / boardSize

        if len(nextCapsules) < len(capsules):
            features["eats-capsule"] = 1.0

        closestCapsuleDist = _closestMazeDistance(nextPos, nextCapsules, walls)
        if closestCapsuleDist is not None:
            features["closest-capsule"] = closestCapsuleDist / boardSize

        activeGhostPositions = []
        scaredGhostPositions = []
        for ghostState in ghostStates:
            ghostPos = _nearestPoint(ghostState.getPosition())
            if ghostState.scaredTimer > 2:
                scaredGhostPositions.append(ghostPos)
            else:
                activeGhostPositions.append(ghostPos)

        closestActiveGhost = _closestMazeDistance(nextPos, activeGhostPositions, walls)
        if closestActiveGhost is not None:
            # 가까운 위험 유령은 큰 패널티가 되도록 역수 feature를 씁니다.
            features["active-ghost-inverse"] = 1.0 / (closestActiveGhost + 1.0)
            if closestActiveGhost <= 1:
                features["active-ghost-1-step"] = 1.0
            if closestActiveGhost <= 2:
                features["active-ghost-2-step"] = 1.0

            # 막다른 길에서 유령이 가까우면 특히 위험합니다.
            legalNextActions = successor.getLegalActions(0)
            if Directions.STOP in legalNextActions:
                legalNextActions.remove(Directions.STOP)
            if len(legalNextActions) <= 1 and closestActiveGhost <= 4:
                features["dead-end-danger"] = 1.0

            # Pacman이 움직인 직후, 다음 유령 턴에서 잡힐 확률을 대략 계산합니다.
            # RandomGhost 기준으로 위험한 유령 행동의 비율이 높을수록 피하게 됩니다.
            killMoves = 0
            totalGhostMoves = 0
            for ghostIndex, ghostState in enumerate(successor.getGhostStates(), start=1):
                if ghostState.scaredTimer > 0:
                    continue
                for ghostAction in successor.getLegalActions(ghostIndex):
                    totalGhostMoves += 1
                    if successor.generateSuccessor(ghostIndex, ghostAction).isLose():
                        killMoves += 1
            if totalGhostMoves > 0:
                features["ghost-kill-prob"] = float(killMoves) / totalGhostMoves

        closestScaredGhost = _closestMazeDistance(nextPos, scaredGhostPositions, walls)
        if closestScaredGhost is not None:
            # 먹을 수 있는 유령은 가까울수록 좋습니다.
            features["scared-ghost-inverse"] = 1.0 / (closestScaredGhost + 1.0)

        # 위험 유령이 가까울 때는 캡슐 방향으로 가는 행동을 더 가치 있게 봅니다.
        if closestActiveGhost is not None and closestActiveGhost <= 5 and closestCapsuleDist is not None:
            features["capsule-when-danger"] = 1.0 / (closestCapsuleDist + 1.0)

        return features


class CompetitionQAgent(ApproximateQAgent):
    """
    경쟁용 Pacman agent입니다.

    일반 ApproximateQAgent처럼 feature와 weight로 Q(s,a)를 계산하지만,
    처음부터 쓸 만한 weight를 넣어 두었습니다. 그래서 학습 게임이 적어도
    음식 쪽으로 가고, 위험 유령을 피하고, 캡슐/먹을 수 있는 유령을 챙깁니다.
    """

    def __init__(self, extractor='CompetitionExtractor', epsilon=0.0, gamma=0.8,
                 alpha=0.0, numTraining=0, depth='auto', **args):
        ApproximateQAgent.__init__(
            self,
            extractor=extractor,
            epsilon=epsilon,
            gamma=gamma,
            alpha=alpha,
            numTraining=numTraining,
            **args
        )
        if depth == 'auto':
            self.searchDepth = None
        else:
            self.searchDepth = int(depth)

        # weight가 클수록 그 feature가 행동 선택에 더 큰 영향을 줍니다.
        # 양수 weight는 그 feature가 큰 행동을 선호하고, 음수 weight는 피합니다.
        self.weights["bias"] = 0.0
        self.weights["score-change"] = 5.0
        self.weights["wins"] = 1000.0
        self.weights["loses"] = -1000.0
        self.weights["stops"] = -80.0
        self.weights["reverses"] = -5.0
        self.weights["eats-food"] = 35.0
        self.weights["food-left"] = -120.0
        self.weights["closest-food"] = -70.0
        self.weights["eats-capsule"] = 80.0
        self.weights["closest-capsule"] = -20.0
        self.weights["capsule-when-danger"] = 70.0
        self.weights["active-ghost-inverse"] = -220.0
        self.weights["active-ghost-1-step"] = -350.0
        self.weights["active-ghost-2-step"] = -120.0
        self.weights["ghost-kill-prob"] = -500.0
        self.weights["dead-end-danger"] = -250.0
        self.weights["scared-ghost-inverse"] = 120.0

    def getAction(self, state):
        """
        경쟁용 행동 선택입니다.

        기본 Q-learning처럼 Q값만 바로 보지 않고, Pacman이 한 번 움직인 뒤
        유령들이 랜덤으로 움직이는 상황까지 몇 수 앞서 평균 내 봅니다.
        그래서 작은 맵에서 유령에게 바로 잡히는 행동을 더 잘 피합니다.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None

        scores = []
        depth = self._getSearchDepth(state)
        for action in legalActions:
            successor = state.generatePacmanSuccessor(action)
            value = self._expectimaxValue(successor, 1, depth)
            scores.append((value, action))

        bestValue = max(value for value, action in scores)
        bestActions = [action for value, action in scores if value == bestValue]
        action = random.choice(bestActions)
        self.doAction(state, action)
        return action

    def _getSearchDepth(self, state):
        """
        작은 맵은 depth 3도 빠르고 안정적입니다.
        큰 맵은 depth 3이 너무 느려질 수 있어서 depth 2를 기본으로 씁니다.
        """
        if self.searchDepth is not None:
            return self.searchDepth

        walls = state.getWalls()
        boardSize = walls.width * walls.height
        numGhosts = state.getNumAgents() - 1
        if boardSize <= 80 and numGhosts <= 1:
            return 3
        return 2

    def _expectimaxValue(self, state, agentIndex, depth):
        """
        Pacman은 가장 좋은 행동을 고른다고 보고(max),
        유령은 RandomGhost처럼 가능한 행동을 평균낸다고 봅니다(expectation).
        """
        if state.isWin() or state.isLose():
            return self._evaluateState(state)

        if agentIndex == 0 and depth <= 0:
            return self._evaluateState(state)

        numAgents = state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        if len(legalActions) == 0:
            return self._evaluateState(state)

        if agentIndex == 0:
            return max(
                self._expectimaxValue(state.generateSuccessor(agentIndex, action), 1, depth)
                for action in legalActions
            )

        nextAgent = agentIndex + 1
        nextDepth = depth
        if nextAgent == numAgents:
            nextAgent = 0
            nextDepth = depth - 1

        values = [
            self._expectimaxValue(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
            for action in legalActions
        ]
        return sum(values) / float(len(values))

    def _evaluateState(self, state):
        """
        lookahead가 멈춘 지점의 상태 점수입니다.
        실제 게임 score에 음식/유령/캡슐 거리를 더해서 더 안전한 상태를 선호합니다.
        """
        score = state.getScore()
        if state.isWin():
            return score + 10000.0
        if state.isLose():
            return score - 100000.0

        walls = state.getWalls()
        compactBoard = walls.width * walls.height <= 160
        pacmanPos = state.getPacmanPosition()
        foodList = state.getFood().asList()
        capsules = state.getCapsules()

        score -= 8.0 * len(foodList)
        score -= 25.0 * len(capsules)

        closestFood = _closestMazeDistance(pacmanPos, foodList, walls)
        if closestFood is not None:
            score -= 2.0 * closestFood

        closestCapsule = _closestMazeDistance(pacmanPos, capsules, walls)
        if closestCapsule is not None:
            score -= 1.0 * closestCapsule

        activeGhostPositions = []
        scaredGhostPositions = []
        for ghostState in state.getGhostStates():
            ghostPos = _nearestPoint(ghostState.getPosition())
            if ghostState.scaredTimer > 2:
                scaredGhostPositions.append(ghostPos)
            else:
                activeGhostPositions.append(ghostPos)

        activeGhostDist = _closestMazeDistance(pacmanPos, activeGhostPositions, walls)
        if activeGhostDist is not None:
            if activeGhostDist <= 1:
                score -= 10000.0
            elif activeGhostDist <= 2:
                score -= 1500.0
            elif compactBoard and activeGhostDist <= 3:
                score -= 700.0
            elif compactBoard and activeGhostDist <= 5:
                score -= 200.0
            else:
                score -= 250.0 / activeGhostDist

            # 위험 유령이 가까울 때는 캡슐까지의 거리를 훨씬 중요하게 봅니다.
            if compactBoard and closestCapsule is not None and activeGhostDist <= 8:
                score -= 8.0 * closestCapsule

            legalActions = state.getLegalActions(0)
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)
            if compactBoard and len(legalActions) <= 1 and activeGhostDist <= 5:
                score -= 700.0

        scaredGhostDist = _closestMazeDistance(pacmanPos, scaredGhostPositions, walls)
        if scaredGhostDist is not None:
            score += 200.0 / (scaredGhostDist + 1.0)

        return score
