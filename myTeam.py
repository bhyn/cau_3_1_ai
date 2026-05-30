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


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
  # The submitted team is one attacker and one defender by default.
  return [globals()[first](firstIndex), globals()[second](secondIndex)]


class TeamAgent(CaptureAgent):
  """Shared helpers for a fast reflex capture team."""

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    # Cache map information that is reused every turn.  This keeps chooseAction
    # cheap enough for the one-second time limit.
    self.start = gameState.getAgentPosition(self.index)
    self.walls = gameState.getWalls()
    self.width = self.walls.width
    self.height = self.walls.height
    self.midX = self.width // 2
    self.homeX = self.midX - 1 if self.red else self.midX
    self.enemyX = self.midX if self.red else self.midX - 1
    self.initialFood = len(self.getFood(gameState).asList())

    # Boundary cells are the legal crossings between our side and enemy side.
    # They are useful for returning home and for defensive patrol.
    self.homeBoundary = self._openColumn(self.homeX)
    self.enemyBoundary = self._openColumn(self.enemyX)
    self.legalPositions = [
      (x, y)
      for x in range(self.width)
      for y in range(self.height)
      if not self.walls[x][y]
    ]

    # Dead ends are risky when an active enemy ghost is visible nearby, so the
    # offensive agent gives those positions an extra penalty.
    self.deadEnds = self._computeDeadEnds()
    self.patrolTarget = self._selectPatrolTarget()

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)

    # Standing still almost never helps in capture, but keep it if it is the
    # only legal action.
    if len(actions) > 1 and Directions.STOP in actions:
      actions.remove(Directions.STOP)

    # This is a one-step reflex policy: simulate each legal move, score the
    # successor state with hand-tuned features, then choose among the best moves.
    scores = [(self.evaluateAction(gameState, action), action) for action in actions]
    bestScore = max(score for score, action in scores)
    bestActions = [action for score, action in scores if score == bestScore]
    return random.choice(bestActions)

  def evaluateAction(self, gameState, action):
    raise NotImplementedError

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()

    # Berkeley capture agents can occasionally be between grid cells.  Evaluating
    # at the next grid point makes distance features more stable.
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

  def _computeDeadEnds(self):
    # Peel off degree-1 corridors until only non-dead-end cells remain.
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
      # Prefer exact maze distance from the precomputed distancer.
      return self.getMazeDistance(pos1, pos2)
    except Exception:
      # Some unusual layouts may pass a non-open cell; fall back instead of
      # crashing during evaluation.
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

    # If a defended food dot disappeared since our last observation, an invader
    # was likely there even if it is no longer visible.
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

    # Only active enemy ghosts are dangerous.  Scared ghosts are handled as
    # possible bonus targets in the offensive evaluation.
    active = self.activeGhosts(successor)
    if not active:
      return 0

    ghostDistances = [self.safeDistance(pos, ghostPos) for idx, state, ghostPos in active]
    minGhost = min(ghostDistances)
    penalty = 0

    # Close ghosts are the main way to lose points, so this penalty dominates
    # ordinary food-distance preferences.
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


class OffensiveAgent(TeamAgent):
  """Food-focused agent that retreats only when visible danger is close."""

  def evaluateAction(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    pos = myState.getPosition()

    # Positive score means good for our team.  Eating food or killing an enemy
    # shows up immediately in scoreDelta.
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

    # Main attacking goal: reduce remaining food and move toward the closest
    # food, with a small bonus for dense nearby food clusters.
    if food:
      foodDistance = self.minDistance(pos, food)
      score -= 6.0 * foodDistance
      score -= 2.0 * len(food)

      nearbyFood = sum(1 for dot in food if self.safeDistance(pos, dot) <= 4)
      score += 6 * nearbyFood

    if capsules:
      capsuleDistance = self.minDistance(pos, capsules)
      # Capsules matter most when an active ghost is close.
      if activeGhostDistance <= 6:
        score -= 5.5 * capsuleDistance
      else:
        score -= 0.7 * capsuleDistance

    if len(self.getCapsules(gameState)) > len(capsules):
      score += 350

    if myState.isPacman and activeGhostDistance <= 5:
      # If a visible ghost is threatening us, prefer paths back to our boundary.
      score -= 7.0 * self.homeDistance(pos)

    for idx, ghostState, ghostPos in self.scaredGhosts(successor):
      distance = self.safeDistance(pos, ghostPos)
      # Chase scared ghosts only when the timer is long enough to reach them.
      if distance < ghostState.scaredTimer - 2:
        score += max(0, 18 - 2 * distance)

    if myState.getPosition() == self.start and gameState.getAgentState(self.index).isPacman:
      score -= 2500

    return score


class DefensiveAgent(TeamAgent):
  """Home-side defender with a late-game/off-score attack fallback."""

  def evaluateAction(self, gameState, action):
    # The second agent normally defends, but can become a second attacker when
    # defense is quiet and the map/score makes attacking worthwhile.
    if self.shouldAttack(gameState):
      return self.attackEvaluation(gameState, action)
    return self.defenseEvaluation(gameState, action)

  def shouldAttack(self, gameState):
    # Visible invaders or newly eaten defended food always take priority.
    if self.visibleInvaders(gameState) or self.recentlyLostFood(gameState):
      return False

    remainingFood = len(self.getFood(gameState).asList())
    if remainingFood <= 5:
      return True

    # Small-food layouts are easier to lose by over-attacking, so keep the
    # defender home unless the game is almost finished.
    if self.initialFood < 35:
      return False

    # Food-rich layouts reward early pressure from both agents.
    if self.initialFood >= 60 and self.getScore(gameState) <= 0:
      return True
    if self.getScore(gameState) <= -4:
      return True
    timeLeft = getattr(gameState.data, 'timeleft', 0)
    return timeLeft and timeLeft < 260 and self.getScore(gameState) <= 0

  def attackEvaluation(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    pos = myState.getPosition()

    # This is a lighter version of the offensive evaluation for the defender's
    # temporary attack mode.
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

    # Defensive mode values staying on our side, catching invaders, and guarding
    # useful central/capsule positions.
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
      # When an invader is visible, chasing it is more important than patrol.
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
      # If an invader was seen indirectly through eaten food, move toward the
      # missing dot to intercept.
      target = self.closestTarget(pos, lostFood)
      score -= 24 * self.safeDistance(pos, target)
      return score

    defendedCapsules = self.getCapsulesYouAreDefending(successor)
    if defendedCapsules:
      score -= 7 * self.minDistance(pos, defendedCapsules)

    score -= 10 * self.safeDistance(pos, self.patrolTarget)
    if pos in self.homeBoundary:
      score += 20
    return score
