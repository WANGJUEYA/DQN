import numpy
from gym import Env
from gym.envs.classic_control import rendering

UNIT = 41  # pixels

LEFT = 'LEFT'
RIGHT = 'RIGHT'
UP = 'UP'
DOWN = 'DOWN'

ACTIONS = [LEFT, RIGHT, UP, DOWN]
ACTIONS_STEP = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}

DEFAULT_MAZE = [
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0],
]


class MazeEnv(Env):

    def __init__(self, maze=None, maze_size=(8, 8)):
        super(MazeEnv, self).__init__()
        # 画板
        self.viewer = None
        if maze is None:
            maze = MazeEnv.random_maze(maze_size)
        else:
            if MazeEnv.check_maze(maze) is False:
                raise Exception("Not Find Path!")
        # 初始化迷宫地图
        self.maze = numpy.array(maze)
        rows, cols = self.maze.shape
        self.rat = (0, 0)
        self.cheese = (rows - 1, cols - 1)  # target cell where the "cheese"

        # 当前走过的路径
        self.visited = None
        # 最小收益
        self.min_reward = None
        # 总收益
        self.total_reward = None
        self.reset()

    def step(self, action):
        if isinstance(action, int):
            action = ACTIONS[action]
        sr, sc = self.rat
        self.visited.append((sr, sc, action))
        row, col = ACTIONS_STEP[action]
        self.rat = (sr + row, sc + col)

        if numpy.array_equal(self.rat, self.cheese):
            reward = 1
            done = True
        else:
            rows, cols = self.maze.shape
            reward = -0.1 / (rows * cols)
            done = False
        info = {}

        return self.rat, reward, done, info

    def reset(self):
        self.rat = (0, 0)
        self.visited = []
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0

    def render(self, mode='human'):
        rows, cols = self.maze.shape
        if self.viewer is None:
            self.viewer = rendering.Viewer((cols + 2) * UNIT, (rows + 2) * UNIT)
        # 画网格
        for i in range(rows + 2):
            self.viewer.draw_line((UNIT, UNIT * i), (cols * UNIT + UNIT, UNIT * i))  # 横线
        for j in range(cols + 2):
            self.viewer.draw_line((UNIT * j, UNIT), (UNIT * j, rows * UNIT + UNIT))  # 竖线

        # 出口，用红色表示出口
        self.viewer.draw_polygon([(0, 0), (0, UNIT - 1), (UNIT - 1, UNIT - 1), (UNIT - 1, 0)], filled=True,
                                 color=(1, 0, 0)).add_attr(
            rendering.Transform((cols * UNIT, UNIT)))
        # 老鼠，用黄色圆圈
        rat_row, rat_col = self.rat
        rat_row_point, rat_col_point = self.render_point_convert(rat_row, rat_col)
        self.viewer.draw_circle(18, color=(0.8, 0.6, 0.4)).add_attr(
            rendering.Transform(
                translation=(rat_row_point + (UNIT / 2), rat_col_point + (UNIT / 2))))
        # 用黑色表示墙
        for i in range(rows):
            for j in range(cols):
                if self.maze[i][j] == 1:
                    self.viewer.draw_polygon([(0, 0), (0, UNIT - 1), (UNIT - 1, UNIT - 1), (UNIT - 1, 0)], filled=True,
                                             color=(0, 0, 0)).add_attr(
                        rendering.Transform(self.render_point_convert(i, j)))
        # 用黄色表示走过的路径
        self.render_visited_with_dashed_line()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_point_convert(self, i, j):
        rows, cols = self.maze.shape
        return (j + 1) * UNIT, (rows - i) * UNIT

    def render_visited_with_dashed_line(self):
        bi, bj = self.render_point_convert(0, 0)
        bi, bj = bi + UNIT / 2, bj + UNIT / 2

        def draw_dashed_line(begin, action_step):
            debugger is True and print(begin, action_step)
            li, lj = begin
            x, y = action_step
            for step in range(int(UNIT / 4)):
                ni, nj = li + 2 * y, lj - 2 * x
                if step % 2 == 0:
                    debugger is True and print((li, lj))
                    self.viewer.draw_line((li, lj), (ni, nj))
                li, lj = ni, nj
            debugger is True and print((li, lj))
            debugger is True and print('----------------------')
            return li, lj

        last = None
        debugger and print(self.visited)
        for item in self.visited:
            i, j, action = item

            self.viewer.draw_polygon([(0, 0), (0, UNIT - 1), (UNIT - 1, UNIT - 1), (UNIT - 1, 0)], filled=True,
                                     color=(1, 1, 0)).add_attr(rendering.Transform(self.render_point_convert(i, j)))
            if last is not None:
                bi, bj = draw_dashed_line((bi, bj), last)
            last = ACTIONS_STEP[action]
            bi, bj = draw_dashed_line((bi, bj), last)

    # 生成一个指定长度的迷宫数组
    @staticmethod
    def random_maze(maze_size):
        def create_maze():
            matrix = []
            rows, cols = maze_size
            for i in range(rows):
                row = []
                for j in range(cols):
                    row.append(numpy.random.choice([0, 1]))
                matrix.append(row)
            matrix[0][0] = 0
            matrix[rows - 1][cols - 1] = 0
            return matrix

        maze = create_maze()
        while MazeEnv.check_maze(maze) is False:
            maze = create_maze()
        print(maze)
        return maze

    # 检查迷宫是否有通路
    @staticmethod
    def check_maze(maze):
        if not maze or len(maze) == 0 or len(maze[0]) == 0:
            return False
        row, col = len(maze), len(maze[0])
        visited = [[False] * col for _ in range(row)]

        def dfs(x, y):
            if maze[x][y] == 0:
                if visited[x][y]:
                    return True
                else:
                    visited[x][y] = True
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < row and 0 <= ny < col:
                            if maze[nx][ny] == 0:
                                dfs(nx, ny)
                    return True
            return False

        # 从左上角开始进行DFS
        dfs(0, 0)
        return visited[row - 1][col - 1]


debugger = True
if __name__ == "__main__":
    # print(MazeEnv.random_maze((2, 6)))
    # env = MazeEnv(DEFAULT_MAZE)
    # env = MazeEnv(None, (2, 6))
    env = MazeEnv()
    env.step(DOWN)
    env.step(RIGHT)
    env.step(RIGHT)
    env.step(RIGHT)
    env.step(DOWN)
    while True:
        env.render()
        debugger = False
