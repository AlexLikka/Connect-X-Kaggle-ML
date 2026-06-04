"""PUCT MCTS guided by a policy-value network."""

import math
import numpy as np
import torch

from rl.connectx import (
    COLUMNS,
    INAROW,
    ROWS,
    board_to_tensor,
    drop_piece,
    is_winning_move,
    opponent,
    valid_moves,
)


class Node:
    def __init__(self, prior=0.0):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def predict(model, board, mark, device):
    planes = torch.from_numpy(board_to_tensor(board, mark)).unsqueeze(0).to(device)
    legal_mask = torch.zeros((1, COLUMNS), dtype=torch.float32, device=device)
    for col in valid_moves(board):
        legal_mask[0, col] = 1.0
    with torch.no_grad():
        logits, value = model(planes, legal_mask=legal_mask)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs, float(value.item())


def add_dirichlet_noise(root, alpha=0.3, frac=0.25):
    if not root.children:
        return
    moves = list(root.children)
    noise = np.random.dirichlet([alpha] * len(moves))
    for move, n in zip(moves, noise):
        child = root.children[move]
        child.prior = child.prior * (1.0 - frac) + float(n) * frac


def expand(node, model, board, mark, device):
    priors, value = predict(model, board, mark, device)
    for move in valid_moves(board):
        if move not in node.children:
            node.children[move] = Node(prior=priors[move])
    return value


def select_child(node, c_puct):
    best_score = -float("inf")
    best_move = None
    best_child = None
    sqrt_visits = math.sqrt(max(1, node.visit_count))
    for move, child in node.children.items():
        ucb = -child.value + c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
        if ucb > best_score:
            best_score = ucb
            best_move = move
            best_child = child
    return best_move, best_child


def run_mcts(model, board, mark, simulations, device="cpu", c_puct=1.5, add_noise=False):
    root = Node()
    expand(root, model, board, mark, device)
    if add_noise:
        add_dirichlet_noise(root)

    for _ in range(simulations):
        node = root
        sim_board = board[:]
        sim_mark = mark
        path = [node]
        terminal_value = None

        while node.children:
            move, node = select_child(node, c_puct)
            sim_board, row = drop_piece(sim_board, move, sim_mark)
            if row >= 0 and is_winning_move(sim_board, row, move, sim_mark, ROWS, COLUMNS, INAROW):
                terminal_value = -1.0
                path.append(node)
                break
            sim_mark = opponent(sim_mark)
            path.append(node)

        if terminal_value is None:
            if not valid_moves(sim_board):
                value = 0.0
            else:
                value = expand(node, model, sim_board, sim_mark, device)
        else:
            value = terminal_value

        # Values alternate perspective on the path back to the root.
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    visits = np.zeros(COLUMNS, dtype=np.float32)
    for move, child in root.children.items():
        visits[move] = child.visit_count
    if visits.sum() <= 0:
        for move in valid_moves(board):
            visits[move] = 1.0
    return visits / visits.sum()


def select_action_from_policy(policy, temperature=1.0):
    if temperature <= 1e-6:
        return int(np.argmax(policy))
    adjusted = np.power(policy, 1.0 / temperature)
    adjusted = adjusted / adjusted.sum()
    return int(np.random.choice(np.arange(len(policy)), p=adjusted))
