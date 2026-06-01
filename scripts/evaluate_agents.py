"""Evaluate local ConnectX agents against baseline opponents."""

import argparse
import os
import sys
import time
from collections import defaultdict

from kaggle_environments import evaluate, make

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import agents.search_agent as search_module
from agents.search_agent import agent as search_agent
from agents.search_agent import rule_agent


AGENTS = {
    "search": search_agent,
    "rule": rule_agent,
    "random": "random",
    "negamax": "negamax",
}


def summarize_rewards(rewards):
    stats = defaultdict(int)
    for mine, theirs in rewards:
        if mine > theirs:
            stats["wins"] += 1
        elif mine < theirs:
            stats["losses"] += 1
        else:
            stats["draws"] += 1
    total = len(rewards)
    score = sum(row[0] for row in rewards) / total if total else 0.0
    return {
        "episodes": total,
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "mean_reward": score,
    }


def run_match(agent_name, opponent_name, episodes):
    subject = AGENTS[agent_name]
    opponent = AGENTS[opponent_name]

    start = time.time()
    first_rewards = []
    second_raw = []
    for episode_index in range(episodes):
        print(
            f"running {agent_name} vs {opponent_name} "
            f"episode {episode_index + 1}/{episodes} (first order)",
            flush=True,
        )
        first_rewards.extend(evaluate("connectx", [subject, opponent], num_episodes=1))
    for episode_index in range(episodes):
        print(
            f"running {agent_name} vs {opponent_name} "
            f"episode {episode_index + 1}/{episodes} (second order)",
            flush=True,
        )
        second_raw.extend(evaluate("connectx", [opponent, subject], num_episodes=1))
    elapsed = time.time() - start

    second_rewards = [[row[1], row[0]] for row in second_raw]
    return summarize_rewards(first_rewards + second_rewards), elapsed


def validate_submission(path, time_limit):
    namespace = {}
    with open(path, "r", encoding="utf-8") as handle:
        code = handle.read()
    exec(compile(code, path, "exec"), namespace)
    if "DEFAULT_TIME_LIMIT" in namespace:
        namespace["DEFAULT_TIME_LIMIT"] = time_limit
    submission_agent = namespace.get("agent")
    if not callable(submission_agent):
        raise RuntimeError(f"{path} does not define a callable agent")

    env = make("connectx", debug=True)
    env.run([submission_agent, submission_agent])
    statuses = [state.status for state in env.state]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"submission self-play failed with statuses: {statuses}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="search", choices=sorted(AGENTS))
    parser.add_argument("--opponents", nargs="+", default=["random", "negamax", "rule"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=0.35,
        help="Per-move search budget for local evaluation. Use 1.65 for submit-like strength.",
    )
    parser.add_argument("--validate-submission", default="submission.py")
    args = parser.parse_args()

    search_module.DEFAULT_TIME_LIMIT = args.time_limit

    if args.validate_submission:
        print(f"validating {args.validate_submission}...", flush=True)
        validate_submission(args.validate_submission, args.time_limit)
        print(f"validated {args.validate_submission}: self-play completed")

    for opponent in args.opponents:
        print(f"starting benchmark vs {opponent}...", flush=True)
        summary, elapsed = run_match(args.agent, opponent, args.episodes)
        print(
            f"{args.agent:>8} vs {opponent:<8} "
            f"episodes={summary['episodes']:>3} "
            f"wins={summary['wins']:>3} "
            f"draws={summary['draws']:>3} "
            f"losses={summary['losses']:>3} "
            f"mean_reward={summary['mean_reward']:.3f} "
            f"elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
