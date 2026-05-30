# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).

from captureAgents import CaptureAgent
from game import Actions, Directions
from util import nearestPoint
import random


# 전체 흐름:
# 1. OffensiveAgent는 상대 진영으로 가서 음식을 많이 먹는다.
# 2. DefensiveAgent는 우리 진영을 지키다가, 안전하면 같이 공격한다.
# 3. 둘 다 매 턴마다 "갈 수 있는 방향"을 하나씩 미리 가본 뒤,
#    가장 점수가 좋아 보이는 방향을 고른다.
def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
  # 기본 팀 구성은 공격수 1명, 수비수 1명이다.
  return [globals()[first](firstIndex), globals()[second](secondIndex)]


class TeamAgent(CaptureAgent):
  """공격수와 수비수가 같이 쓰는 도구 모음."""

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    # 처음 한 번만 지도 정보를 저장해 둔다.
    # 매 턴마다 다시 계산하면 느려지기 때문이다.
    self.start = gameState.getAgentPosition(self.index)
    self.walls = gameState.getWalls()
    self.width = self.walls.width
    self.height = self.walls.height
    self.midX = self.width // 2
    self.homeX = self.midX - 1 if self.red else self.midX
    self.enemyX = self.midX if self.red else self.midX - 1
    self.initialFood = len(self.getFood(gameState).asList())
    self.initialCapsules = len(self.getCapsules(gameState)) + len(self.getCapsulesYouAreDefending(gameState))
    self.isSparseNoCapsuleMap = self.initialFood < 35 and self.initialCapsules == 0

    # 가운데 선 근처에서 실제로 지나갈 수 있는 칸들이다.
    # 공격수는 도망칠 때 여기로 돌아오고, 수비수는 여기 근처를 지킨다.
    self.homeBoundary = self._openColumn(self.homeX)
    self.enemyBoundary = self._openColumn(self.enemyX)
    self.legalPositions = [
      (x, y)
      for x in range(self.width)
      for y in range(self.height)
      if not self.walls[x][y]
    ]

    # 막다른 길은 적 유령이 가까울 때 위험하다.
    # 그래서 공격수는 막다른 길에 들어가는 선택을 더 싫어하게 만든다.
    self.deadEnds = self._computeDeadEnds()
    self.patrolTarget = self._selectPatrolTarget()
    self.patrolTargets = self._selectPatrolTargets()

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)

    # 가만히 있기(STOP)는 대부분 손해다.
    # 단, 정말 움직일 곳이 없으면 남겨 둔다.
    if len(actions) > 1 and Directions.STOP in actions:
      actions.remove(Directions.STOP)

    # 각 방향으로 한 번 가본 척해 보고 점수를 매긴다.
    # 그중 점수가 가장 높은 방향을 고른다.
    scores = [(self.evaluateAction(gameState, action), action) for action in actions]
    bestScore = max(score for score, action in scores)
    bestActions = [action for score, action in scores if score == bestScore]
    return random.choice(bestActions)

  def evaluateAction(self, gameState, action):
    raise NotImplementedError

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()

    # 가끔 칸과 칸 사이에 걸쳐 있을 수 있다.
    # 그런 경우 한 번 더 움직여서 정확한 칸 기준으로 평가한다.
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    return successor

  def _openColumn(self, x):
    return [(x, y) for y in range(1, self.height - 1) if not self.walls[x][y]]

  def _selectPatrolTarget(self):
    if not self.homeBoundary:
      return self.start
    centerY = self.height // 2
    return min(self.homeBoundary, key=lambda p: abs(p[1] - centerY))

  def _selectPatrolTargets(self):
    if not self.homeBoundary:
      return [self.start]
    lowToHigh = sorted(self.homeBoundary, key=lambda p: p[1])
    center = self._selectPatrolTarget()
    patrols = [center]
    if lowToHigh[0] != center:
      patrols.append(lowToHigh[0])
    if lowToHigh[-1] != center:
      patrols.append(lowToHigh[-1])
    return patrols

  def _computeDeadEnds(self):
    # 막다른 길 찾기:
    # 길이 하나뿐인 끝 칸부터 지워 나가면 위험한 막다른 통로가 나온다.
    degrees = {}
    neighbors = {}
    for pos in self.legalPositions:
      adj = [n for n in Actions.getLegalNeighbors(pos, self.walls)]
      neighbors[pos] = adj
      degrees[pos] = len(adj)

    queue = [pos for pos, degree in degrees.items() if degree <= 1]
    dead = set()
    while queue:
      pos = queue.pop()
      if pos in dead:
        continue
      dead.add(pos)
      for nxt in neighbors[pos]:
        if nxt not in dead:
          degrees[nxt] -= 1
          if degrees[nxt] <= 1:
            queue.append(nxt)
    return dead

  def safeDistance(self, pos1, pos2):
    if pos1 is None or pos2 is None:
      return 9999
    try:
      # 벽을 돌아가는 실제 미로 거리를 우선 사용한다.
      return self.getMazeDistance(pos1, pos2)
    except Exception:
      # 이상한 좌표가 들어와도 게임이 멈추지 않게 대략 거리로 대신 계산한다.
      return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

  def minDistance(self, pos, targets):
    if not targets:
      return 9999
    return min(self.safeDistance(pos, target) for target in targets)

  def closestTarget(self, pos, targets):
    if not targets:
      return None
    return min(targets, key=lambda target: self.safeDistance(pos, target))

  def scoreDelta(self, gameState, successor):
    return self.getScore(successor) - self.getScore(gameState)

  def isHome(self, pos):
    if pos is None:
      return False
    if self.red:
      return pos[0] < self.width / 2
    return pos[0] >= self.width / 2

  def visibleEnemies(self, gameState):
    enemies = []
    for idx in self.getOpponents(gameState):
      state = gameState.getAgentState(idx)
      pos = state.getPosition()
      if pos is not None:
        enemies.append((idx, state, pos))
    return enemies

  def visibleInvaders(self, gameState):
    return [
      (idx, state, pos)
      for idx, state, pos in self.visibleEnemies(gameState)
      if state.isPacman
    ]

  def visibleGhosts(self, gameState):
    return [
      (idx, state, pos)
      for idx, state, pos in self.visibleEnemies(gameState)
      if not state.isPacman
    ]

  def activeGhosts(self, gameState):
    return [
      (idx, state, pos)
      for idx, state, pos in self.visibleGhosts(gameState)
      if state.scaredTimer <= 2
    ]

  def scaredGhosts(self, gameState):
    return [
      (idx, state, pos)
      for idx, state, pos in self.visibleGhosts(gameState)
      if state.scaredTimer > 2
    ]

  def recentlyLostFood(self, gameState):
    previous = self.getPreviousObservation()
    if previous is None:
      return []

    # 방금 전에는 있던 우리 음식이 사라졌다면,
    # 안 보이더라도 적 팩맨이 그 근처에 있었다고 생각한다.
    oldFood = set(self.getFoodYouAreDefending(previous).asList())
    newFood = set(self.getFoodYouAreDefending(gameState).asList())
    return list(oldFood - newFood)

  def reversePenalty(self, gameState, action, amount):
    direction = gameState.getAgentState(self.index).configuration.direction
    if action == Directions.REVERSE[direction]:
      return -amount
    return 0

  def dangerPenalty(self, gameState, successor, pos):
    myState = successor.getAgentState(self.index)
    if not myState.isPacman:
      return 0

    # 공격 중일 때 무섭지 않은 적 유령만 진짜 위험하다.
    # 겁먹은 유령은 오히려 잡을 수 있으므로 여기서는 위험으로 보지 않는다.
    active = self.activeGhosts(successor)
    if not active:
      return 0

    ghostDistances = [self.safeDistance(pos, ghostPos) for idx, state, ghostPos in active]
    minGhost = min(ghostDistances)
    penalty = 0

    # 적 유령이 가까우면 음식을 조금 더 먹는 것보다 도망치는 것이 중요하다.
    # 그래서 가까울수록 큰 벌점을 준다.
    if minGhost <= 1:
      penalty -= 12000
    elif minGhost == 2:
      penalty -= 2600
    elif minGhost == 3:
      penalty -= 900
    elif minGhost == 4:
      penalty -= 250

    if minGhost <= 5 and pos in self.deadEnds:
      penalty -= 650
    return penalty

  def homeDistance(self, pos):
    return self.minDistance(pos, self.homeBoundary)

  def currentPatrolTarget(self):
    # 음식이 적고 통로가 긴 맵에서는 한곳에 서 있으면 반대쪽이 뚫린다.
    # 그래서 가운데, 아래쪽, 위쪽 입구를 천천히 번갈아 보게 한다.
    if self.isSparseNoCapsuleMap and self.patrolTargets:
      index = (len(self.observationHistory) // 35) % len(self.patrolTargets)
      return self.patrolTargets[index]
    return self.patrolTarget


class OffensiveAgent(TeamAgent):
  """주로 상대 음식을 먹는 공격수."""

  def evaluateAction(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    pos = myState.getPosition()

    # 점수가 올라가는 행동은 좋은 행동이다.
    # 음식을 먹거나 적을 잡으면 scoreDelta가 커진다.
    score = 0
    score += 1200 * self.scoreDelta(gameState, successor)
    score += 8 * self.getScore(successor)
    score += self.reversePenalty(gameState, action, 8)

    if action == Directions.STOP:
      score -= 80

    score += self.dangerPenalty(gameState, successor, pos)

    food = self.getFood(successor).asList()
    capsules = self.getCapsules(successor)
    activeGhosts = self.activeGhosts(successor)
    activeGhostDistance = self.minDistance(pos, [p for i, s, p in activeGhosts])
    invaders = self.visibleInvaders(successor)

    if invaders and not myState.isPacman:
      # 상대가 둘 다 공격하면 수비수 혼자서는 막기 어렵다.
      # 공격수도 우리 진영에 있을 때는 잠깐 적 팩맨을 같이 쫓는다.
      invaderDistance = self.minDistance(pos, [p for i, s, p in invaders])
      score -= 85 * invaderDistance
      score -= 450 * len(invaders)
      if invaderDistance <= 2:
        score += 700

    # 공격수의 기본 목표는 가까운 음식을 향해 가는 것이다.
    # 주변에 음식이 여러 개 있으면 그쪽이 더 좋아 보이게 한다.
    if food:
      foodDistance = self.minDistance(pos, food)
      score -= 6.0 * foodDistance
      score -= 2.0 * len(food)

      nearbyFood = sum(1 for dot in food if self.safeDistance(pos, dot) <= 4)
      score += 6 * nearbyFood

    if capsules:
      capsuleDistance = self.minDistance(pos, capsules)
      # 적 유령이 가까울 때는 캡슐이 특히 중요하다.
      if activeGhostDistance <= 6:
        score -= 5.5 * capsuleDistance
      else:
        score -= 0.7 * capsuleDistance

    if len(self.getCapsules(gameState)) > len(capsules):
      score += 350

    if myState.isPacman and activeGhostDistance <= 5:
      # 적 유령이 가까우면 무리하지 말고 우리 진영 쪽으로 돌아가게 한다.
      score -= 7.0 * self.homeDistance(pos)

    for idx, ghostState, ghostPos in self.scaredGhosts(successor):
      distance = self.safeDistance(pos, ghostPos)
      # 겁먹은 유령은 시간이 충분할 때만 쫓아간다.
      if distance < ghostState.scaredTimer - 2:
        score += max(0, 18 - 2 * distance)

    if myState.getPosition() == self.start and gameState.getAgentState(self.index).isPacman:
      score -= 2500

    return score


class DefensiveAgent(TeamAgent):
  """주로 우리 진영을 지키는 수비수."""

  def evaluateAction(self, gameState, action):
    # 평소에는 수비를 한다.
    # 하지만 적이 안 보이고 공격이 더 이득이면 잠깐 공격수처럼 움직인다.
    if self.shouldAttack(gameState):
      return self.attackEvaluation(gameState, action)
    return self.defenseEvaluation(gameState, action)

  def shouldAttack(self, gameState):
    # 적 팩맨이 보이거나 우리 음식이 방금 먹혔다면 무조건 수비한다.
    if self.visibleInvaders(gameState) or self.recentlyLostFood(gameState):
      return False

    remainingFood = len(self.getFood(gameState).asList())
    if remainingFood <= 5:
      return True

    if self.isSparseNoCapsuleMap and self.getScore(gameState) <= -2:
      return True

    timeLeft = getattr(gameState.data, 'timeleft', 0)
    if self.isSparseNoCapsuleMap and timeLeft < 900 and self.getScore(gameState) <= 0:
      return True

    # 음식이 적은 맵에서는 한 번 뚫리면 손해가 크다.
    # 그래서 수비수를 쉽게 공격에 보내지 않는다.
    if self.initialFood < 35:
      return False

    # 음식이 많은 맵에서는 둘이 같이 공격하면 점수를 빨리 벌 수 있다.
    if self.initialFood >= 60 and self.getScore(gameState) <= 0:
      return True
    if self.getScore(gameState) <= -4:
      return True

    # 기본 맵에서 계속 0:0으로 굳으면 리그 승점이 부족하다.
    # 침입자가 안 보이는 중후반에는 수비수도 공격에 가서 승리를 노린다.
    return timeLeft and timeLeft < 1900 and self.getScore(gameState) <= 0

  def attackEvaluation(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    pos = myState.getPosition()

    # 수비수가 임시로 공격할 때 쓰는 점수 계산이다.
    # 공격수보다 조금 더 조심스럽게 움직인다.
    score = 0
    score += 1100 * self.scoreDelta(gameState, successor)
    score += 7 * self.getScore(successor)
    score += self.reversePenalty(gameState, action, 6)
    score += self.dangerPenalty(gameState, successor, pos)

    food = self.getFood(successor).asList()
    if food:
      score -= 5.0 * self.minDistance(pos, food)
      score -= 1.5 * len(food)

    activeGhostDistance = self.minDistance(pos, [p for i, s, p in self.activeGhosts(successor)])
    if myState.isPacman and activeGhostDistance <= 4:
      score -= 9.0 * self.homeDistance(pos)

    if action == Directions.STOP:
      score -= 90
    return score

  def defenseEvaluation(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    pos = myState.getPosition()

    # 수비 모드에서는 세 가지를 중요하게 본다.
    # 1. 우리 진영에 있기
    # 2. 적 팩맨 잡기
    # 3. 가운데나 캡슐 근처 지키기
    score = 0
    score += 1300 * self.scoreDelta(gameState, successor)
    score += 8 * self.getScore(successor)
    score += self.reversePenalty(gameState, action, 4)

    if action == Directions.STOP:
      score -= 120

    if myState.isPacman:
      score -= 1800
    else:
      score += 250

    invaders = self.visibleInvaders(successor)
    if invaders:
      # 적 팩맨이 보이면 순찰보다 추격이 먼저다.
      distances = [self.safeDistance(pos, invaderPos) for idx, state, invaderPos in invaders]
      closest = min(distances)
      score -= 95 * closest
      score -= 700 * len(invaders)
      if myState.scaredTimer > 0:
        score += 160 * max(0, 5 - closest)
      else:
        score += 260 * max(0, 5 - closest)
      return score

    lostFood = self.recentlyLostFood(gameState)
    if lostFood:
      # 적이 안 보여도 우리 음식이 사라졌다면 그쪽으로 가서 막는다.
      target = self.closestTarget(pos, lostFood)
      score -= 24 * self.safeDistance(pos, target)
      return score

    defendedCapsules = self.getCapsulesYouAreDefending(successor)
    if defendedCapsules:
      score -= 7 * self.minDistance(pos, defendedCapsules)

    score -= 10 * self.safeDistance(pos, self.currentPatrolTarget())
    if pos in self.homeBoundary:
      score += 20
    return score
