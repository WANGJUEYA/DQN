import threading

import numpy
from gymnasium import Env, spaces, logger
import pygame

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
        self.pygame_initialized = False
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
        # 当前状态 - 修改为向量格式 (x, y)
        self.observation_space = spaces.Box(low=0, high=max(rows, cols), shape=(2,), dtype=numpy.float32)

        # 初始化起点和终点
        self.rat = (0, 0)
        self.cheese = (rows - 1, cols - 1)  # target cell where the "cheese"

        # 当前走过的路径
        self.visited = []
        # 总步数
        self.steps_beyond_done = 0
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
            reward = 100.0
        else:
            if move:
                self.steps_beyond_done += 1
                # 如果进行了移动，有少量惩罚，促使机器人找到"最短路径"
                reward = -0.01 / (rows * cols)
                # 如果走到移动过的位置，加大惩罚; 每次增加 0.3 的惩罚
                for item in self.visited:
                    i, j, _ = item
                    if (i, j) == (nr, nc):
                        reward -= 0.03
                        if reward < -0.9:
                            break
            else:
                # 不移动进行较多的惩罚
                reward = -0.1
        
        # 返回状态向量而不是元组
        state_vector = numpy.array([self.rat[0], self.rat[1]], dtype=numpy.float32)
        return state_vector, reward, done, {}

    def reset(self):
        self.rat = (0, 0)
        self.visited = []
        self.steps_beyond_done = 0
        # 返回状态向量而不是元组
        state_vector = numpy.array([self.rat[0], self.rat[1]], dtype=numpy.float32)
        return state_vector

    def render(self, mode='human'):
        rows, cols = self.maze.shape
        
        # 初始化pygame
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True
            
        if self.viewer is None:
            self.viewer = pygame.display.set_mode(((cols + 2) * UNIT, (rows + 2) * UNIT))
            pygame.display.set_caption("Maze Environment")
        
        # 清空屏幕
        self.viewer.fill((255, 255, 255))  # 白色背景
        
        # 画网格
        for i in range(rows + 2):
            pygame.draw.line(self.viewer, (0, 0, 0), (UNIT, UNIT * i), (cols * UNIT + UNIT, UNIT * i))  # 横线
        for j in range(cols + 2):
            pygame.draw.line(self.viewer, (0, 0, 0), (UNIT * j, UNIT), (UNIT * j, rows * UNIT + UNIT))  # 竖线

        # 出口，用红色表示出口
        exit_x, exit_y = self.render_point_convert(rows - 1, cols - 1)
        pygame.draw.rect(self.viewer, (255, 0, 0), (exit_x, exit_y, UNIT, UNIT))
        
        # 用黑色表示墙
        for i in range(rows):
            for j in range(cols):
                if self.maze[i][j] == 1:
                    wall_x, wall_y = self.render_point_convert(i, j)
                    pygame.draw.rect(self.viewer, (0, 0, 0), (wall_x, wall_y, UNIT, UNIT))
        
        # 用黄色表示走过的路径
        self.render_visited_with_dashed_line()
        
        # 老鼠，用黄色圆圈
        rat_row, rat_col = self.rat
        rat_x, rat_y = self.render_point_convert(rat_row, rat_col)
        pygame.draw.circle(self.viewer, (255, 255, 0), (rat_x + UNIT // 2, rat_y + UNIT // 2), UNIT // 2 - 5)

        pygame.display.flip()

    def render_point_convert(self, i, j):
        """将迷宫坐标转换为屏幕坐标"""
        rows, cols = self.maze.shape
        # 修正坐标转换：i是行（从上到下），j是列（从左到右）
        screen_x = (j + 1) * UNIT
        screen_y = (i + 1) * UNIT
        return screen_x, screen_y

    def render_visited_with_dashed_line(self):
        if self.viewer is None or not self.visited:
            return
            
        # 绘制访问过的路径，用虚线连接单元格中心
        if len(self.visited) > 0:
            # 从起点开始
            start_pos = (0, 0)
            
            for item in self.visited:
                i, j, action = item
                end_pos = (i, j)
                
                # 获取起点和终点的屏幕坐标（中心点）
                start_x, start_y = self.render_point_convert(start_pos[0], start_pos[1])
                start_x += UNIT // 2
                start_y += UNIT // 2
                
                end_x, end_y = self.render_point_convert(end_pos[0], end_pos[1])
                end_x += UNIT // 2
                end_y += UNIT // 2
                
                # 绘制从起点到终点的虚线（细灰色）
                self.draw_dashed_line(start_x, start_y, end_x, end_y, color=(128, 128, 128), line_width=1)
                
                # 更新起点为当前终点
                start_pos = end_pos
            
            # 绘制当前老鼠位置到上一个位置的连线（细黑色实线）
            if len(self.visited) > 0:
                last_item = self.visited[-1]
                last_i, last_j, _ = last_item
                
                # 获取上一个位置和当前位置的中心点
                last_x, last_y = self.render_point_convert(last_i, last_j)
                last_x += UNIT // 2
                last_y += UNIT // 2
                
                current_x, current_y = self.render_point_convert(self.rat[0], self.rat[1])
                current_x += UNIT // 2
                current_y += UNIT // 2
                
                # 绘制当前移动的连线（细黑色实线）
                pygame.draw.line(self.viewer, (0, 0, 0), 
                               (int(last_x), int(last_y)), 
                               (int(current_x), int(current_y)), 1)

    def draw_dashed_line(self, start_x, start_y, end_x, end_y, dash_length=8, gap_length=4, color=(0, 0, 0), line_width=1):
        """绘制虚线"""
        if self.viewer is None:
            return
            
        # 计算线段长度和方向
        dx = end_x - start_x
        dy = end_y - start_y
        distance = (dx**2 + dy**2)**0.5
        
        if distance == 0:
            return
            
        # 单位向量
        unit_x = dx / distance
        unit_y = dy / distance
        
        # 绘制虚线
        current_x, current_y = start_x, start_y
        remaining_distance = distance
        
        while remaining_distance > 0:
            # 计算当前段的长度
            segment_length = min(dash_length, remaining_distance)
            
            # 计算段终点
            segment_end_x = current_x + unit_x * segment_length
            segment_end_y = current_y + unit_y * segment_length
            
            # 绘制线段
            pygame.draw.line(self.viewer, color, 
                           (int(current_x), int(current_y)), 
                           (int(segment_end_x), int(segment_end_y)), line_width)
            
            # 移动到段终点
            current_x = segment_end_x
            current_y = segment_end_y
            remaining_distance -= segment_length
            
            # 跳过间隙
            if remaining_distance > gap_length:
                current_x += unit_x * gap_length
                current_y += unit_y * gap_length
                remaining_distance -= gap_length
            else:
                break

    def close(self):
        if self.viewer:
            pygame.display.quit()
            self.viewer = None
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

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
        logger.deprecation("maze: %s", maze)
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
