# Planning and MaxEnt IRL. A path has PATH_LENGTH cells and must end on the goal,
# so both can be solved exactly by going backwards from the last step.
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from flex.config import GRID_SIZE, PATH_LENGTH
from flex.world import move_table

N_CELLS = GRID_SIZE * GRID_SIZE
CAN_MOVE = move_table()

def next_cell_values(values, blocked):
    # row a, column b: the value of moving from a to b, or `blocked` if b can't be reached from a
    return np.where(CAN_MOVE, values, blocked)

def last_step_values(reward, goal, blocked):
    # the path has to end on the goal
    values = np.full(N_CELLS, blocked)
    values[goal] = reward[goal]
    return values

def best_path(reward, start, goal, worst=False):
    # the path with the highest (or lowest) total reward, and that total
    blocked = np.inf if worst else -np.inf
    value = np.zeros((PATH_LENGTH, N_CELLS))
    next_cell = np.zeros((PATH_LENGTH, N_CELLS), dtype=int)
    value[-1] = last_step_values(reward, goal, blocked)
    for t in range(PATH_LENGTH - 2, -1, -1):
        options = next_cell_values(value[t + 1], blocked)
        if worst:
            next_cell[t] = options.argmin(axis=1)
            value[t] = reward + options.min(axis=1)
        else:
            next_cell[t] = options.argmax(axis=1)
            value[t] = reward + options.max(axis=1)

    path = [start]
    for t in range(PATH_LENGTH - 1):
        path.append(int(next_cell[t, path[-1]]))
    return path, value[0, start]

def soft_values(reward, goal):
    # like best_path, but a soft max (log-sum-exp) over the next cells instead of the max
    value = np.zeros((PATH_LENGTH, N_CELLS))
    value[-1] = last_step_values(reward, goal, -np.inf)
    for t in range(PATH_LENGTH - 2, -1, -1):
        value[t] = reward + logsumexp(next_cell_values(value[t + 1], -np.inf), axis=1)
    return value

def move_probabilities(value, t):
    # probs[a, b]: chance of moving from a to b at step t when p(path) ~ exp(reward)
    options = next_cell_values(value[t + 1], -np.inf)
    total = logsumexp(options, axis=1, keepdims=True)
    total[np.isinf(total)] = 0  # cells that can't reach the goal in time; their probabilities stay 0
    return np.exp(options - total)

def expected_visits(reward, start, goal, value=None):
    # visits[t, cell]: chance of being in that cell at step t
    if value is None:
        value = soft_values(reward, goal)
    visits = np.zeros((PATH_LENGTH, N_CELLS))
    visits[0, start] = 1.0
    for t in range(PATH_LENGTH - 1):
        visits[t + 1] = visits[t] @ move_probabilities(value, t)
    return visits

def sample_demos(reward, start, goal, n, rng):
    value = soft_values(reward, goal)
    probs = [move_probabilities(value, t) for t in range(PATH_LENGTH - 1)]
    demos = []
    for _ in range(n):
        path = [start]
        for t in range(PATH_LENGTH - 1):
            path.append(int(rng.choice(N_CELLS, p=probs[t][path[-1]])))
        demos.append(path)
    return demos

def learn_weights(table, demos, start, goal, l2=1e-3):
    # MaxEnt IRL: the weights that make the demos most likely. Returns (weights, log-likelihood).
    n_features = table.shape[1]
    if n_features == 0:
        return np.zeros(0), -len(demos) * soft_values(np.zeros(N_CELLS), goal)[0, start]

    # standardize the features so L-BFGS behaves; the weights are scaled back at the end
    mean, spread = table.mean(axis=0), table.std(axis=0)
    spread[spread < 1e-9] = 1.0
    scaled = (table - mean) / spread

    demo_counts = np.zeros(n_features)
    for demo in demos:
        demo_counts += scaled[demo].sum(axis=0)
    demo_counts /= len(demos)

    def loss_and_gradient(weights):
        reward = scaled @ weights
        value = soft_values(reward, goal)
        expected_counts = expected_visits(reward, start, goal, value).sum(axis=0) @ scaled
        log_likelihood = weights @ demo_counts - value[0, start]
        loss = -log_likelihood + l2 * weights @ weights
        gradient = -(demo_counts - expected_counts) + 2 * l2 * weights
        return loss, gradient

    result = minimize(loss_and_gradient, np.zeros(n_features), jac=True, method='L-BFGS-B')
    weights = result.x
    log_likelihood = len(demos) * -(result.fun - l2 * weights @ weights)
    return weights / spread, float(log_likelihood)

def path_reward(reward, path):
    total = 0.0
    for cell in path:
        total += reward[cell]
    return total

def normalized_reward(reward, path, start, goal):
    # (R - worst) / (best - worst) under the true reward, so 1.00 is the best path
    _, best = best_path(reward, start, goal)
    _, worst = best_path(reward, start, goal, worst=True)
    if best <= worst:
        return 1.0
    return float((path_reward(reward, path) - worst) / (best - worst))
