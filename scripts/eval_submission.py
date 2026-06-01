import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
# ensure local kaggle-environments package is importable
sys.path.insert(0, str(root / 'kaggle-environments-0.1.4'))
from kaggle_environments import make
import submission_ml
from agents import search_agent

def run_eval(episodes=100, opponent='negamax'):
    wins=draws=losses=0
    rewards=[]
    # resolve opponent name to callable
    if opponent == 'negamax' or opponent == 'search':
        opp_callable = search_agent.agent
    elif opponent == 'rule':
        opp_callable = search_agent.rule_agent
    elif opponent == 'random':
        # simple random agent
        def rand_agent(observation, configuration):
            import random
            board = list(observation.board)
            cols = configuration.columns
            moves = [c for c in range(cols) if board[c] == 0]
            return random.choice(moves) if moves else 0
        opp_callable = rand_agent
    else:
        raise ValueError(f'unknown opponent: {opponent}')

    for i in range(episodes):
        env = make('connectx', debug=False)
        env.run([submission_ml.agent, opp_callable])
        r = env.state[0].reward
        rewards.append(r)
        if r>0: wins+=1
        elif r==0: draws+=1
        else: losses+=1
        print(f"episode {i+1}/{episodes} reward={r}")
    import statistics
    print('\nSUMMARY')
    print('episodes=', episodes)
    print('wins=', wins)
    print('draws=', draws)
    print('losses=', losses)
    print('mean_reward=', statistics.mean(rewards))

if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--opponent', type=str, default='negamax')
    args = p.parse_args()
    run_eval(args.episodes, args.opponent)
