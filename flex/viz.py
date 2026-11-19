import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from flex.config import DISTANCE_CAP, GRID_SIZE
from flex.world import index_to_cell, list_objects

OBJECT_COLORS = {'stove': '#d62728', 'kettle': '#ff7f0e', 'fridge': '#1f77b4', 'toaster': '#bcbd22',
                 'lava': '#d62728', 'swamp': '#8c564b', 'meadow': '#2ca02c', 'quicksand': '#bcbd22'}
GOAL_COLOR = '#2ca02c'

def draw_objects(ax, env, truth):
    for object_type, properties, positions in list_objects(env):
        color = OBJECT_COLORS.get(object_type, '#555555')
        for x, y in positions:
            # shade the cap radius around the objects the true reward avoids
            if truth in properties or truth == object_type:
                ax.add_patch(Circle((x, y), DISTANCE_CAP, color=color, alpha=0.08, lw=0))
            ax.add_patch(Circle((x, y), DISTANCE_CAP, fill=False, ls='--', lw=0.8, color=color, alpha=0.5))
            ax.add_patch(FancyBboxPatch((x - 0.38, y - 0.38), 0.76, 0.76, color=color, zorder=3,
                                        boxstyle='round,pad=0.02,rounding_size=0.15'))
            ax.text(x, y + 0.02, object_type[:2].upper(), ha='center', va='center', color='white',
                    fontsize=8, weight='bold', zorder=4)
            ax.text(x, y + 0.62, object_type, ha='center', va='center', fontsize=6.5, color=color, zorder=4)

def draw_layout(ax, env, paths, title, truth):
    # paths: list of (cells, label, color, line style)
    ax.set_facecolor('#fafafa')
    for k in range(GRID_SIZE + 1):
        ax.axhline(k - 0.5, color='#dddddd', lw=0.7)
        ax.axvline(k - 0.5, color='#dddddd', lw=0.7)
    draw_objects(ax, env, truth)
    start_x, start_y = env['starts'][0]
    goal_x, goal_y = env['goals_pos'][0]
    ax.plot(start_x, start_y, 'o', ms=15, color='black', zorder=5)
    ax.text(start_x, start_y, 'S', color='white', ha='center', va='center', fontsize=8, weight='bold', zorder=6)
    ax.plot(goal_x, goal_y, '*', ms=22, color=GOAL_COLOR, mec='white', zorder=5)

    for i, (path, label, color, style) in enumerate(paths):
        xy = np.array([index_to_cell(cell) for cell in path], float)
        offset = (i - (len(paths) - 1) / 2) * 0.1  # so overlapping paths stay visible
        ax.plot(xy[:, 0] + offset, xy[:, 1] + offset, style, color=color, lw=3, alpha=0.9, label=label, zorder=7)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7.5, frameon=False)
    ax.set_title(title, fontsize=10, weight='bold')
