import numpy
from gym import Env

UNIT = 40  # pixels

LEFT = 'LEFT'
RIGHT = 'RIGHT'
UP = 'UP'
DOWN = 'DOWN'

ACTIONS = [LEFT, RIGHT, UP, DOWN]
ACTIONS_STEP = {
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    UP: (0, -1),
    DOWN: (0, 1),
}


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
        self.visited.add(self.rat)
        sr, sc = self.rat
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
        self.visited = set()
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        rows, cols = self.maze.shape
        if self.viewer is None:
            self.viewer = rendering.Viewer((cols + 2) * UNIT, (rows + 2) * UNIT)
        # 画网格
        for i in range(rows + 2):
            self.viewer.draw_line((UNIT, UNIT * i), (cols * UNIT + UNIT, UNIT * i))  # 横线
        for j in range(cols + 2):
            self.viewer.draw_line((UNIT * j, UNIT), (UNIT * j, rows * UNIT + UNIT))  # 竖线

        def view_point(i, j):
            return (j + 1) * UNIT, (rows - i) * UNIT

        # 出口，用红色表示出口
        self.viewer.draw_polygon([(0, 0), (0, UNIT), (UNIT, UNIT), (UNIT, 0)], filled=True,
                                 color=(1, 0, 0)).add_attr(
            rendering.Transform((cols * UNIT, UNIT)))
        # 老鼠，用黄色圆圈
        rat_row, rat_col = self.rat
        rat_row_point, rat_col_point = view_point(rat_row, rat_col)
        self.viewer.draw_circle(18, color=(0.8, 0.6, 0.4)).add_attr(
            rendering.Transform(
                translation=(rat_row_point + (UNIT / 2), rat_col_point + (UNIT / 2))))
        # 用黑色表示墙
        for i in range(rows):
            for j in range(cols):
                if self.maze[i][j] == 1:
                    self.viewer.draw_polygon([(0, 0), (0, UNIT), (UNIT, UNIT), (UNIT, 0)], filled=True,
                                             color=(0, 0, 0)).add_attr(
                        rendering.Transform(view_point(i, j)))
        # 用黄色表示走过的路径
        for item in self.visited:
            i, j = item
            self.viewer.draw_polygon([(0, 0), (0, UNIT), (UNIT, UNIT), (UNIT, 0)], filled=True,
                                     color=(1, 1, 0)).add_attr(
                rendering.Transform(view_point(i, j)))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
                        if 0 <= nx < row and 0 <= ny < col and maze[nx][ny] == 0 and dfs(nx, ny):
                            return True
            return False

        # 从左上角开始进行DFS
        dfs(0, 0)
        return visited[row - 1][col - 1]


if __name__ == "__main__":
    # print(MazeEnv.random_maze((2, 6)))
    # env = MazeEnv(None, (2, 6))
    env = MazeEnv()
    env.step(DOWN)
    while True:
        env.render()
