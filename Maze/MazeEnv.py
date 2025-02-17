import threading

import numpy
from gym import Env, spaces, logger
from gym.envs.classic_control import rendering

# 当迷宫尺寸过大时，可以缩小像素，并且取消唯一通路检查！numpy.random.choice([0, 0, 0, 1])
UNIT = 41  # pixels

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

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

        # 初始化行为选项
        self.action_space = spaces.Discrete(4)
        # 当前状态
        self.observation_space = 2

        # 初始化起点和终点
        self.rat = (0, 0)
        self.cheese = (rows - 1, cols - 1)  # target cell where the "cheese"

        # 当前走过的路径
        self.visited = None
        # 总步数
        self.steps_beyond_done = None
        self.reset()

    def step(self, action):
        if isinstance(action, int):
            action = ACTIONS[action]
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert action in ACTIONS, err_msg
        rows, cols = self.maze.shape
        x, y = ACTIONS_STEP[action]
        sr, sc = self.rat
        nr, nc = sr + x, sc + y

        # 是否移动
        move = False
        # 如果不是边界或不是障碍，执行下一个
        if 0 <= nr < rows and 0 <= nc < cols and self.maze[nr][nc] == 0:
            move = True
            self.rat = (nr, nc)
            self.visited.append((sr, sc, action))

        # 游戏是否结束
        done = numpy.array_equal(self.rat, self.cheese)
        if done:
            # 如果成功了，返回最大收益
            reward = 1.0
        else:
            if move:
                self.steps_beyond_done += 1
                # 如果进行了移动，有少量惩罚，促使机器人找到“最短路径”
                reward = -0.1 / (rows * cols)
                # 如果走到移动过的位置，加大惩罚; 每次增加 0.3 的惩罚
                for item in self.visited:
                    i, j, _ = item
                    if (i, j) == (nr, nc):
                        reward -= 0.3
                        if reward < -0.9:
                            break
            else:
                # 不移动进行较多的惩罚
                reward = -1
        return self.rat, reward, done, {}

    def reset(self):
        self.rat = (0, 0)
        self.visited = []
        self.steps_beyond_done = 0
        return self.rat

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
        # 用黑色表示墙
        for i in range(rows):
            for j in range(cols):
                if self.maze[i][j] == 1:
                    self.viewer.draw_polygon([(0, 0), (0, UNIT - 1), (UNIT - 1, UNIT - 1), (UNIT - 1, 0)], filled=True,
                                             color=(0, 0, 0)).add_attr(
                        rendering.Transform(self.render_point_convert(i, j)))
        # 用黄色表示走过的路径
        self.render_visited_with_dashed_line()
        # 老鼠，用黄色圆圈
        rat_row, rat_col = self.rat
        rat_row_point, rat_col_point = self.render_point_convert(rat_row, rat_col)
        self.viewer.draw_circle((UNIT - 5) / 2, color=(0.8, 0.6, 0.4)).add_attr(
            rendering.Transform(
                translation=(rat_row_point + (UNIT / 2), rat_col_point + (UNIT / 2))))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_point_convert(self, i, j):
        rows, cols = self.maze.shape
        return (j + 1) * UNIT, (rows - i) * UNIT

    def render_visited_with_dashed_line(self):
        bi, bj = self.render_point_convert(0, 0)
        bi, bj = bi + UNIT / 2, bj + UNIT / 2

        def draw_dashed_line(begin, action_step):
            li, lj = begin
            x, y = action_step
            for step in range(int(UNIT / 4)):
                ni, nj = li + 2 * y, lj - 2 * x
                if step % 2 == 0:
                    self.viewer.draw_line((li, lj), (ni, nj))
                li, lj = ni, nj
            return li, lj

        last = None
        exist = set()
        for item in self.visited:
            i, j, action = item
            find = (i, j) in exist
            # 没有走过才用黄色覆盖
            if find is not True:
                self.viewer.draw_polygon(
                    [(0, 0), (0, UNIT - 1), (UNIT - 1, UNIT - 1), (UNIT - 1, 0)], filled=True,
                    color=(1, 1, 0)).add_attr(rendering.Transform(self.render_point_convert(i, j)))
            exist.add((i, j))

            if last is not None:
                bi, bj = draw_dashed_line((bi, bj), last)
            last = ACTIONS_STEP[action]
            bi, bj = draw_dashed_line((bi, bj), last)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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
        logger.info("maze: %s", maze)
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


if __name__ == "__main__":
    # env = MazeEnv(DEFAULT_MAZE)
    # env = MazeEnv(None, (2, 6))
    env = MazeEnv()

    # env.step(DOWN)
    # env.step(RIGHT)
    # env.step(RIGHT)
    # env.step(RIGHT)
    # env.step(DOWN)
    action = None
    close = False


    def on_key():
        global close
        global action
        global input_thread
        while close is not True:
            input_text = input("w|s|a|d 选择下一步方向:")
            logger.info(f'你按下了键：{input_text}')
            if input_text == 'q':
                close = True
                break
            elif input_text == 'w':
                action = UP
            elif input_text == 's':
                action = DOWN
            elif input_text == 'a':
                action = LEFT
            elif input_text == 'd':
                action = RIGHT
            else:
                action = None


    input_thread = threading.Thread(target=on_key)
    input_thread.start()

    while close is not True:
        if action is not None:
            _, _, done, _ = env.step(action)
            if done is True:
                close = True
                break
            action = None
        env.render()

    env.close()
