"""Strong pure-search baseline agents for Kaggle ConnectX.

The public entry point is `agent(observation, configuration)`, matching the
Kaggle submission interface. The implementation intentionally avoids external
dependencies so that the same logic can be copied into `submission.py`.
"""

import math
import random
import time


EMPTY = 0
WIN_SCORE = 1_000_000
DEFAULT_TIME_LIMIT = 1.65
MAX_SEARCH_DEPTH = 8
TIME_MARGIN = 0.25


def opponent(mark):
    return 1 if mark == 2 else 2


def valid_moves(board, columns):
    return [c for c in range(columns) if board[c] == EMPTY]


def ordered_columns(columns):
    center = columns // 2
    return sorted(range(columns), key=lambda c: abs(c - center))


def next_open_row(board, column, rows, columns):
    for row in range(rows - 1, -1, -1):
        if board[row * columns + column] == EMPTY:
            return row
    return -1


def drop_piece(board, column, mark, rows, columns):
    row = next_open_row(board, column, rows, columns)
    if row < 0:
        return None, -1
    next_board = board[:]
    next_board[row * columns + column] = mark
    return next_board, row


def is_winning_move(board, row, column, mark, rows, columns, inarow):
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dr, dc in directions:
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


def find_winning_move(board, mark, rows, columns, inarow):
    for column in ordered_columns(columns):
        if board[column] != EMPTY:
            continue
        next_board, row = drop_piece(board, column, mark, rows, columns)
        if row >= 0 and is_winning_move(next_board, row, column, mark, rows, columns, inarow):
            return column
    return None


def window_score(window, mark, inarow):
    opp = opponent(mark)
    mine = window.count(mark)
    theirs = window.count(opp)
    empty = window.count(EMPTY)

    if mine and theirs:
        return 0
    if mine == inarow:
        return WIN_SCORE
    if theirs == inarow:
        return -WIN_SCORE
    if mine:
        if mine == inarow - 1 and empty == 1:
            return 900
        if mine == inarow - 2 and empty == 2:
            return 80
        return mine * mine
    if theirs:
        if theirs == inarow - 1 and empty == 1:
            return -1_200
        if theirs == inarow - 2 and empty == 2:
            return -120
        return -(theirs * theirs)
    return 0


def evaluate_board(board, mark, rows, columns, inarow):
    score = 0
    center = columns // 2
    center_values = [board[row * columns + center] for row in range(rows)]
    score += center_values.count(mark) * 8
    score -= center_values.count(opponent(mark)) * 8

    # Horizontal windows.
    for row in range(rows):
        base = row * columns
        for col in range(columns - inarow + 1):
            score += window_score(board[base + col:base + col + inarow], mark, inarow)

    # Vertical windows.
    for col in range(columns):
        for row in range(rows - inarow + 1):
            window = [board[(row + i) * columns + col] for i in range(inarow)]
            score += window_score(window, mark, inarow)

    # Positive-slope diagonals.
    for row in range(rows - inarow + 1):
        for col in range(columns - inarow + 1):
            window = [board[(row + i) * columns + col + i] for i in range(inarow)]
            score += window_score(window, mark, inarow)

    # Negative-slope diagonals.
    for row in range(inarow - 1, rows):
        for col in range(columns - inarow + 1):
            window = [board[(row - i) * columns + col + i] for i in range(inarow)]
            score += window_score(window, mark, inarow)

    return score


def score_move_for_ordering(board, column, mark, rows, columns, inarow):
    next_board, row = drop_piece(board, column, mark, rows, columns)
    if row < 0:
        return -math.inf
    if is_winning_move(next_board, row, column, mark, rows, columns, inarow):
        return WIN_SCORE
    opp_win = find_winning_move(next_board, opponent(mark), rows, columns, inarow)
    danger = -500_000 if opp_win is not None else 0
    center_bonus = 20 - abs(column - columns // 2) * 3
    return danger + center_bonus + evaluate_board(next_board, mark, rows, columns, inarow)


def order_moves(board, moves, mark, rows, columns, inarow):
    return sorted(
        moves,
        key=lambda c: score_move_for_ordering(board, c, mark, rows, columns, inarow),
        reverse=True,
    )


class SearchTimeout(Exception):
    pass


def negamax(board, mark, depth, alpha, beta, rows, columns, inarow, deadline, cache):
    if time.time() >= deadline:
        raise SearchTimeout

    moves = valid_moves(board, columns)
    if not moves:
        return 0, None

    immediate = find_winning_move(board, mark, rows, columns, inarow)
    if immediate is not None:
        return WIN_SCORE + depth, immediate

    if depth == 0:
        return evaluate_board(board, mark, rows, columns, inarow), None

    original_alpha = alpha
    key = (tuple(board), mark, depth)
    cached = cache.get(key)
    if cached is not None:
        return cached

    best_score = -math.inf
    best_col = moves[0]
    for column in order_moves(board, moves, mark, rows, columns, inarow):
        next_board, row = drop_piece(board, column, mark, rows, columns)
        if row < 0:
            continue
        if is_winning_move(next_board, row, column, mark, rows, columns, inarow):
            score = WIN_SCORE + depth
        else:
            child_score, _ = negamax(
                next_board,
                opponent(mark),
                depth - 1,
                -beta,
                -alpha,
                rows,
                columns,
                inarow,
                deadline,
                cache,
            )
            score = -child_score

        if score > best_score:
            best_score = score
            best_col = column
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    if alpha > original_alpha and best_score < beta:
        cache[key] = (best_score, best_col)
    return best_score, best_col


def choose_action(board, mark, rows, columns, inarow, time_limit=DEFAULT_TIME_LIMIT):
    moves = valid_moves(board, columns)
    if not moves:
        return 0

    winning = find_winning_move(board, mark, rows, columns, inarow)
    if winning is not None:
        return winning

    blocking = find_winning_move(board, opponent(mark), rows, columns, inarow)
    if blocking is not None:
        return blocking

    deadline = time.time() + time_limit
    best_col = ordered_columns(columns)[0]
    for col in ordered_columns(columns):
        if col in moves:
            best_col = col
            break

    cache = {}
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        try:
            _, column = negamax(
                board,
                mark,
                depth,
                -math.inf,
                math.inf,
                rows,
                columns,
                inarow,
                deadline,
                cache,
            )
            if column is not None and column in moves:
                best_col = column
        except SearchTimeout:
            break

    return best_col if best_col in moves else random.choice(moves)


def agent(observation, configuration):
    board = list(observation.board)
    mark = observation.mark
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    timeout = getattr(configuration, "timeout", None)
    time_limit = DEFAULT_TIME_LIMIT
    if isinstance(timeout, (int, float)):
        time_limit = min(time_limit, max(0.10, timeout - TIME_MARGIN))
    return choose_action(board, mark, rows, columns, inarow, time_limit=time_limit)


def rule_agent(observation, configuration):
    board = list(observation.board)
    mark = observation.mark
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    moves = valid_moves(board, columns)
    if not moves:
        return 0

    winning = find_winning_move(board, mark, rows, columns, inarow)
    if winning is not None:
        return winning

    blocking = find_winning_move(board, opponent(mark), rows, columns, inarow)
    if blocking is not None:
        return blocking

    for col in ordered_columns(columns):
        if col in moves:
            return col
    return random.choice(moves)
