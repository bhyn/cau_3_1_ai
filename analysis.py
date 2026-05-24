# analysis.py
# -----------
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


###########################
# ANALYSIS QUESTIONS (Q2) #
###########################

# Set the given parameters to obtain the specified policies through
# value iteration.

# 기본적으로 close exit을 선호하면 경로가 짧고, 먼 exit을 선호하면 경로가 긴가봄 -> 리빙리워드로 너무 길어지지 않게 조절해야 함

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    # discount를 낮게 잡으면 가까운 +1 출구를 더 선호합니다.
    # noise를 0으로 두면 미끄러질 위험이 없어서 cliff 근처 짧은 길을 택할 수 있습니다.
    # living reward를 음수로 두면 오래 돌아다니는 것보다 빨리 끝내는 쪽이 유리합니다.
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    # 가까운 출구를 원하므로 discount는 낮게 둡니다.
    # noise를 주면 cliff 옆 길이 위험해져서 안전한 위쪽 경로를 선택하게 됩니다.
    # living reward는 음수라서 그래도 가까운 출구로 빨리 가려 합니다.
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    # discount를 높게 잡으면 멀리 있는 +10 보상도 충분히 가치 있게 봅니다.
    # noise가 0이면 cliff 근처로 가도 실제로 미끄러지지 않으므로 짧은 위험 경로를 택합니다.
    # living reward를 음수로 둬서 가능한 빨리 +10 출구에 도착하게 합니다.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    # 먼 +10 출구를 선호하도록 discount는 높게 둡니다.
    # noise가 있으면 cliff 옆 길의 기대값이 나빠지므로 안전한 위쪽 경로를 고릅니다.
    # 이건 결국 장기적인 보상을 보면서 안전한 길을 고르는 건데, 그러면 리빙 리워드를 음수로 해놔야 그나마 빨리 +10 출구로 가려고 합니다. 
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    # 매 step마다 양수 보상을 받게 하면 종료하지 않고 계속 움직이는 것이 유리합니다.
    # discount도 충분히 높게 둬서 미래의 living reward를 크게 평가하게 합니다.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
