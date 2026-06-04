"""Stronger pure-search ConnectX submission.

This keeps the Kaggle submission self-contained and dependency-free. It borrows
the useful ideas from stronger board-game search engines: tactical pattern
ordering, fail-soft alpha-beta, principal variation search, and bound-aware
transposition table entries.
"""

import math
import random
import time


EMPTY = 0
WIN_SCORE = 1_000_000
DEFAULT_TIME_LIMIT = 1.65
MAX_SEARCH_DEPTH = 9
TIME_MARGIN = 0.22
EXACT = 0
LOWER = 1
UPPER = 2


def opponent(mark):
    return 1 if mark == 2 else 2


def valid_moves(board, columns):
    return [col for col in range(columns) if board[col] == EMPTY]


def ordered_columns(columns):
    center = columns // 2
    return sorted(range(columns), key=lambda col: abs(col - center))


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


def find_winning_move(board, mark, rows, columns, inarow):
    for column in ordered_columns(columns):
        if board[column] != EMPTY:
            continue
        next_board, row = drop_piece(board, column, mark, rows, columns)
        if row >= 0 and is_winning_move(next_board, row, column, mark, rows, columns, inarow):
            return column
    return None


def count_winning_moves(board, mark, rows, columns, inarow):
    count = 0
    for column in range(columns):
        if board[column] != EMPTY:
            continue
        next_board, row = drop_piece(board, column, mark, rows, columns)
        if row >= 0 and is_winning_move(next_board, row, column, mark, rows, columns, inarow):
            count += 1
    return count


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
            return 1_200
        if mine == inarow - 2 and empty == 2:
            return 90
        return mine * mine
    if theirs:
        if theirs == inarow - 1 and empty == 1:
            return -1_600
        if theirs == inarow - 2 and empty == 2:
            return -140
        return -(theirs * theirs)
    return 0


def iter_windows(board, rows, columns, inarow):
    for row in range(rows):
        base = row * columns
        for col in range(columns - inarow + 1):
            yield board[base + col:base + col + inarow]
    for col in range(columns):
        for row in range(rows - inarow + 1):
            yield [board[(row + i) * columns + col] for i in range(inarow)]
    for row in range(rows - inarow + 1):
        for col in range(columns - inarow + 1):
            yield [board[(row + i) * columns + col + i] for i in range(inarow)]
    for row in range(inarow - 1, rows):
        for col in range(columns - inarow + 1):
            yield [board[(row - i) * columns + col + i] for i in range(inarow)]


def evaluate_board(board, mark, rows, columns, inarow):
    score = 0
    opp = opponent(mark)
    center = columns // 2
    center_values = [board[row * columns + center] for row in range(rows)]
    score += center_values.count(mark) * 10
    score -= center_values.count(opp) * 10

    for window in iter_windows(board, rows, columns, inarow):
        score += window_score(window, mark, inarow)

    my_threats = count_winning_moves(board, mark, rows, columns, inarow)
    opp_threats = count_winning_moves(board, opp, rows, columns, inarow)
    if my_threats >= 2:
        score += 450_000
    elif my_threats == 1:
        score += 35_000
    if opp_threats >= 2:
        score -= 520_000
    elif opp_threats == 1:
        score -= 45_000
    return score


def score_move_for_ordering(board, column, mark, rows, columns, inarow, killer=None):
    next_board, row = drop_piece(board, column, mark, rows, columns)
    if row < 0:
        return -math.inf
    if is_winning_move(next_board, row, column, mark, rows, columns, inarow):
        return 10 * WIN_SCORE

    opp = opponent(mark)
    opp_threats = count_winning_moves(next_board, opp, rows, columns, inarow)
    my_threats = count_winning_moves(next_board, mark, rows, columns, inarow)
    score = evaluate_board(next_board, mark, rows, columns, inarow)
    if opp_threats >= 2:
        score -= 2_000_000
    elif opp_threats == 1:
        score -= 700_000
    if my_threats >= 2:
        score += 1_200_000
    elif my_threats == 1:
        score += 200_000
    score += 35 - abs(column - columns // 2) * 5
    if killer == column:
        score += 80_000
    return score


def order_moves(board, moves, mark, rows, columns, inarow, killer=None):
    return sorted(
        moves,
        key=lambda col: score_move_for_ordering(board, col, mark, rows, columns, inarow, killer),
        reverse=True,
    )


class SearchTimeout(Exception):
    pass


def negamax(board, mark, depth, alpha, beta, rows, columns, inarow, deadline, table, killers, ply):
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

    alpha_orig = alpha
    key = (tuple(board), mark, depth)
    entry = table.get(key)
    if entry is not None:
        entry_depth, entry_score, entry_flag, entry_move = entry
        if entry_depth >= depth:
            if entry_flag == EXACT:
                return entry_score, entry_move
            if entry_flag == LOWER:
                alpha = max(alpha, entry_score)
            elif entry_flag == UPPER:
                beta = min(beta, entry_score)
            if alpha >= beta:
                return entry_score, entry_move
    else:
        entry_move = None

    killer = killers.get(ply, entry_move)
    ordered = order_moves(board, moves, mark, rows, columns, inarow, killer)
    if entry_move in ordered:
        ordered.remove(entry_move)
        ordered.insert(0, entry_move)

    best_score = -math.inf
    best_col = ordered[0]
    first = True

    for column in ordered:
        next_board, row = drop_piece(board, column, mark, rows, columns)
        if row < 0:
            continue
        if is_winning_move(next_board, row, column, mark, rows, columns, inarow):
            score = WIN_SCORE + depth
        elif first:
            child_score, _ = negamax(
                next_board, opponent(mark), depth - 1, -beta, -alpha,
                rows, columns, inarow, deadline, table, killers, ply + 1,
            )
            score = -child_score
        else:
            child_score, _ = negamax(
                next_board, opponent(mark), depth - 1, -alpha - 1, -alpha,
                rows, columns, inarow, deadline, table, killers, ply + 1,
            )
            score = -child_score
            if alpha < score < beta:
                child_score, _ = negamax(
                    next_board, opponent(mark), depth - 1, -beta, -alpha,
                    rows, columns, inarow, deadline, table, killers, ply + 1,
                )
                score = -child_score

        first = False
        if score > best_score:
            best_score = score
            best_col = column
        if score > alpha:
            alpha = score
        if alpha >= beta:
            killers[ply] = column
            break

    if best_score <= alpha_orig:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    else:
        flag = EXACT
    table[key] = (depth, best_score, flag, best_col)
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
    best_col = moves[0]
    for col in ordered_columns(columns):
        if col in moves:
            best_col = col
            break

    table = {}
    killers = {}
    previous_score = 0
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        try:
            window = 80_000 if depth >= 5 and abs(previous_score) < WIN_SCORE // 2 else math.inf
            if window == math.inf:
                score, column = negamax(
                    board, mark, depth, -math.inf, math.inf,
                    rows, columns, inarow, deadline, table, killers, 0,
                )
            else:
                alpha = previous_score - window
                beta = previous_score + window
                score, column = negamax(
                    board, mark, depth, alpha, beta,
                    rows, columns, inarow, deadline, table, killers, 0,
                )
                if score <= alpha or score >= beta:
                    score, column = negamax(
                        board, mark, depth, -math.inf, math.inf,
                        rows, columns, inarow, deadline, table, killers, 0,
                    )
            if column is not None and column in moves:
                best_col = column
                previous_score = score
        except SearchTimeout:
            break

    return best_col if best_col in moves else random.choice(moves)


def agent(observation, configuration):
    board = list(observation.board)
    timeout = getattr(configuration, "timeout", None)
    time_limit = DEFAULT_TIME_LIMIT
    if isinstance(timeout, (int, float)):
        time_limit = min(time_limit, max(0.10, timeout - TIME_MARGIN))
    return choose_action(
        board,
        observation.mark,
        configuration.rows,
        configuration.columns,
        configuration.inarow,
        time_limit=time_limit,
    )
