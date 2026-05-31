"""Run local ConnectX matches between two Kaggle-style submission agents."""

import argparse
import os
import random
import sys
import traceback
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace


EMPTY = 0

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@dataclass
class LoadedAgent:
    name: str
    func: callable


def opponent(mark):
    return 1 if mark == 2 else 2


def random_agent(observation, configuration):
    legal = [column for column in range(configuration.columns) if observation.board[column] == EMPTY]
    return random.choice(legal) if legal else 0


def ordered_columns(columns):
    center = columns // 2
    return sorted(range(columns), key=lambda column: abs(column - center))


def next_open_row(board, column, rows, columns):
    for row in range(rows - 1, -1, -1):
        if board[row * columns + column] == EMPTY:
            return row
    return -1


def drop_piece(board, column, mark, rows, columns):
    row = next_open_row(board, column, rows, columns)
    if row < 0:
        return -1
    board[row * columns + column] = mark
    return row


def is_winning_move(board, row, column, mark, rows, columns, inarow):
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r = row + sign * dr
            c = column + sign * dc
            while 0 <= r < rows and 0 <= c < columns and board[r * columns + c] == mark:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= inarow:
            return True
    return False


def format_board(board, rows, columns):
    tokens = {0: ".", 1: "X", 2: "O"}
    line = "+" + "+".join(["---"] * columns) + "+"
    parts = [line]
    for row in range(rows):
        start = row * columns
        values = [tokens[board[start + column]] for column in range(columns)]
        parts.append("| " + " | ".join(values) + " |")
        parts.append(line)
    parts.append("  " + "   ".join(str(column) for column in range(columns)))
    return "\n".join(parts)


def normalize_action(action):
    if isinstance(action, bool):
        return None
    if isinstance(action, Integral):
        return int(action)
    return None


def load_submission_agent(path, time_limit):
    namespace = {}
    with open(path, "r", encoding="utf-8") as handle:
        code = handle.read()
    exec(compile(code, path, "exec"), namespace)
    if "DEFAULT_TIME_LIMIT" in namespace:
        namespace["DEFAULT_TIME_LIMIT"] = time_limit
    agent = namespace.get("agent")
    if not callable(agent):
        raise RuntimeError(f"{path} does not define a callable agent(observation, configuration)")
    return LoadedAgent(name=os.path.basename(path), func=agent)


def load_agent(spec, time_limit):
    builtin_agents = {
        "random": LoadedAgent(name="random", func=random_agent),
    }
    if spec in builtin_agents:
        return builtin_agents[spec]
    path = os.path.abspath(spec)
    return load_submission_agent(path, time_limit)


def play_single_game(agent_one, agent_two, config, render=False):
    board = [EMPTY] * (config.rows * config.columns)
    agents = {1: agent_one, 2: agent_two}
    move_history = []

    while True:
        mark = 1 if len(move_history) % 2 == 0 else 2
        active = agents[mark]
        observation = SimpleNamespace(board=board[:], mark=mark)

        try:
            raw_action = active.func(observation, config)
        except Exception:
            return {
                "winner": opponent(mark),
                "rewards": {mark: 0.0, opponent(mark): 1.0},
                "statuses": {mark: "ERROR", opponent(mark): "DONE"},
                "board": board,
                "moves": move_history,
                "error": traceback.format_exc(),
            }

        column = normalize_action(raw_action)
        if column is None or column < 0 or column >= config.columns or board[column] != EMPTY:
            return {
                "winner": opponent(mark),
                "rewards": {mark: 0.0, opponent(mark): 1.0},
                "statuses": {mark: "INVALID", opponent(mark): "DONE"},
                "board": board,
                "moves": move_history,
                "error": f"{active.name} returned invalid action: {raw_action!r}",
            }

        row = drop_piece(board, column, mark, config.rows, config.columns)
        move_history.append(column)

        if render:
            print(f"\nTurn {len(move_history)}: {active.name} (mark={mark}) -> column {column}")
            print(format_board(board, config.rows, config.columns))

        if is_winning_move(board, row, column, mark, config.rows, config.columns, config.inarow):
            return {
                "winner": mark,
                "rewards": {mark: 1.0, opponent(mark): 0.0},
                "statuses": {1: "DONE", 2: "DONE"},
                "board": board,
                "moves": move_history,
                "error": None,
            }

        if all(cell != EMPTY for cell in board):
            return {
                "winner": 0,
                "rewards": {1: 0.5, 2: 0.5},
                "statuses": {1: "DONE", 2: "DONE"},
                "board": board,
                "moves": move_history,
                "error": None,
            }


def update_summary(summary, agent_one_mark, result):
    if result["winner"] == 0:
        summary["draws"] += 1
        return
    if result["winner"] == agent_one_mark:
        summary["agent1_wins"] += 1
    else:
        summary["agent2_wins"] += 1


def main():
    parser = argparse.ArgumentParser(description="Run local ConnectX matches between two submission agents.")
    parser.add_argument("agent1", help="Path to the first submission file, or 'random'.")
    parser.add_argument("agent2", help="Path to the second submission file, or 'random'.")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--alternate-first", action="store_true", help="Alternate who plays first across games.")
    parser.add_argument("--render", action="store_true", help="Print the board after every move.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=1.65, help="Injected into DEFAULT_TIME_LIMIT when present.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    config = SimpleNamespace(
        rows=args.rows,
        columns=args.columns,
        inarow=args.inarow,
        timeout=args.time_limit,
    )

    loaded_agent1 = load_agent(args.agent1, args.time_limit)
    loaded_agent2 = load_agent(args.agent2, args.time_limit)

    summary = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}

    for game_index in range(args.games):
        if args.alternate_first and game_index % 2 == 1:
            first = loaded_agent2
            second = loaded_agent1
            agent_one_mark = 2
        else:
            first = loaded_agent1
            second = loaded_agent2
            agent_one_mark = 1

        if args.render:
            print(f"\n=== Game {game_index + 1} ===")
            print(f"{first.name} is mark 1, {second.name} is mark 2")
            print(format_board([EMPTY] * (args.rows * args.columns), args.rows, args.columns))

        result = play_single_game(first, second, config, render=args.render)
        update_summary(summary, agent_one_mark, result)

        winner_name = "draw"
        if result["winner"] == 1:
            winner_name = first.name
        elif result["winner"] == 2:
            winner_name = second.name

        print(
            f"game={game_index + 1} first={first.name} second={second.name} "
            f"winner={winner_name} moves={len(result['moves'])} "
            f"statuses=({result['statuses'][1]}, {result['statuses'][2]})"
        )
        if result["error"]:
            print(result["error"])

    print(
        f"\nSummary: {loaded_agent1.name} wins={summary['agent1_wins']} "
        f"{loaded_agent2.name} wins={summary['agent2_wins']} draws={summary['draws']}"
    )


if __name__ == "__main__":
    main()
