# The environment is the map. It knows the map, tracks the agents
# position, responds to actions, and rewards.
# per my understanding this is the map of the lake so a list that hold the position of start, frozen, hole and goal

# The 8x8 map. Each character is one cell.
#   S = Start    (row 0, col 0)
#   F = Frozen   (safe to walk on)
#   H = Hole     (agent falls in the episode over, no reward)
#   G = Goal     (agent wins episode over, reward = +1)

#map of the lake 
MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
]

# Grid dimensions
GRID_SIZE = 8
NUM_STATES = GRID_SIZE * GRID_SIZE   # 64 states total
NUM_ACTIONS = 4                       # Left, Down, Right, Up

# Actions that can be taken 
LEFT  = 0
DOWN  = 1
RIGHT = 2
UP    = 3

# to be used in the rendering of the lake 
ACTION_SYMBOLS = {LEFT: "←", DOWN: "↓", RIGHT: "→", UP: "↑"}


class FrozenLakeEnv:
    """
    A Frozen Lake grid-world environment.

    State is represented as a single integer (0-63):
        state = row * 8 + col
        row   = state // 8
        col   = state % 8

    This makes the Q-table simpler  it becomes a 64 x 4 array.
    """

    def __init__(self):
        # Find the start state by scanning the map for 'S'
        self.start_state = self._find_cell('S')
        self.state = self.start_state

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self):
        """
        Put the agent back at the start. Anytime the agent falls in a hole
        or reaches the goal it resets 
        Returns the starting state (integer 0 to 63).
        """
        self.state = self.start_state
        return self.state

    #handles which action the agent takes whetheer up,down,right,left
    #update the agents location wherever it end up in the info
    # where is the agent at a state, in the hole no reward, in the goal reward of 1, if not in hole or goal no reward and not done
    #info is a dictionary that tells us what cell type we landed on
    def step(self, action):
        """
        Move the agent one step.

        Parameters
        ----------
        action : int  (0=Left, 1=Down, 2=Right, 3=Up)

        Returns
        -------
        next_state : int   — where the agent ended up
        reward     : float — +1 if Goal reached, 0 otherwise
        done       : bool  — True if episode is over (Hole or Goal)
        info       : dict  — extra info (what cell type we landed on)
        """
        row, col = self._state_to_coords(self.state)

        # --- Calculate where the action would take us ---
        if action == LEFT:
            col = max(col - 1, 0)              # can't go past left edge
        elif action == DOWN:
            row = min(row + 1, GRID_SIZE - 1)  # can't go past bottom edge
        elif action == RIGHT:
            col = min(col + 1, GRID_SIZE - 1)  # can't go past right edge
        elif action == UP:
            row = max(row - 1, 0)              # can't go past top edge

        # --- Update state ---
        self.state = self._coords_to_state(row, col)
        cell = MAP[row][col]

        # --- Assign reward and check if episode is over ---
        if cell == 'G':
            reward = 1.0
            done   = True
        elif cell == 'H':
            reward = 0.0
            done   = True
        else:
            reward = 0.0
            done   = False

        info = {"cell": cell}
        return self.state, reward, done, info

    def render(self):
        """
        Print the grid to the console. The agent's current position is
        shown as '*'.
        """
        print(f"\n  Current state: {self.state}  "
              f"(row={self.state // GRID_SIZE}, col={self.state % GRID_SIZE})\n")

        for row in range(GRID_SIZE):
            row_str = ""
            for col in range(GRID_SIZE):
                state = self._coords_to_state(row, col)
                if state == self.state:
                    row_str += " @ "   # agent's position
                else:
                    row_str += f" {MAP[row][col]} "
            print(row_str)
        print()

    #
    def get_state(self):
        """Return the current state as an integer."""
        return self.state

    #Checks whether a given state is a goal or hole
    def is_terminal(self, state=None):
        """
        Return True if the given state is terminal (Hole or Goal).
        If no state is provided, checks the current state.
        """
        if state is None:
            state = self.state
        row, col = self._state_to_coords(state)
        return MAP[row][col] in ('H', 'G')

 
    # Helper methods (
    #row & column to state and state to row & column conversion methods
    # so to know movement i opted for Single integer state indices but to understand better the movement this function 
    # more or less the agent understand (x,y) coordinates but the q table understand single integer state so we need to convert between the two 
    def _state_to_coords(self, state):
        """Convert integer state → (row, col)."""
        return state // GRID_SIZE, state % GRID_SIZE

    def _coords_to_state(self, row, col):
        """Convert (row, col) → integer state."""
        return row * GRID_SIZE + col

    def _find_cell(self, target):
        """Scan the map and return the state index of a target cell ('S', 'G')."""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if MAP[row][col] == target:
                    return self._coords_to_state(row, col)
        raise ValueError(f"Cell '{target}' not found in map")
