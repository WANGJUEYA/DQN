import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.Maze.MazeEnv import MazeEnv, LEFT, RIGHT, UP, DOWN
import pygame

def main():
    print("=== MazeEnv 按键状态测试 ===")
    
    # 初始化pygame
    pygame.init()
    
    # 创建环境
    # env = MazeEnv(DEFAULT_MAZE)
    # env = MazeEnv(None, (2, 6))
    env = MazeEnv()
    print(f"起点: {env.rat}")
    print(f"终点: {env.cheese}")
    
    # 创建窗口
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Maze Key Pressed Test")
    
    # 设置时钟
    clock = pygame.time.Clock()
    
    print("窗口已创建，请点击窗口使其获得焦点")
    print("按 WASD 或方向键移动，Q 退出")
    
    running = True
    frame_count = 0
    done = False
    
    while running:
        frame_count += 1
        
        # 处理事件
        events = pygame.event.get()
        if events:
            print(f"帧 {frame_count} 捕获事件: {[str(e) for e in events]}")
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                print(f"  收到KeyDown事件: {key_name} (代码: {event.key})")
                
                if event.key == pygame.K_q:
                    print("按了Q，退出")
                    running = False
                elif event.key == pygame.K_w or event.key == pygame.K_UP:
                    print("按了W或上箭头，向上移动")
                    old_pos = env.rat
                    state, reward, done, info = env.step(UP)
                    print(f"    从 {old_pos} 移动到 {env.rat}, 奖励: {reward}")
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    print("按了S或下箭头，向下移动")
                    old_pos = env.rat
                    state, reward, done, info = env.step(DOWN)
                    print(f"    从 {old_pos} 移动到 {env.rat}, 奖励: {reward}")
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    print("按了A或左箭头，向左移动")
                    old_pos = env.rat
                    state, reward, done, info = env.step(LEFT)
                    print(f"    从 {old_pos} 移动到 {env.rat}, 奖励: {reward}")
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    print("按了D或右箭头，向右移动")
                    old_pos = env.rat
                    state, reward, done, info = env.step(RIGHT)
                    print(f"    从 {old_pos} 移动到 {env.rat}, 奖励: {reward}")
                else:
                    print(f"    其他按键: {key_name}")
        
        if done:
            print("到达终点！")
            running = False
        
        # 渲染环境
        env.render()
        
        # 控制帧率
        clock.tick(60)
        
        # 每100帧打印一次状态和窗口焦点
        if frame_count % 100 == 0:
            focused = pygame.display.get_active()
            print(f"运行中... 帧: {frame_count}，窗口获得焦点: {focused}")
            if not focused:
                print("[提示] 请用鼠标点击窗口使其获得焦点，否则无法接收键盘输入！")
    
    # 清理
    env.close()
    pygame.quit()
    print("=== 测试结束 ===")

if __name__ == "__main__":
    main() 