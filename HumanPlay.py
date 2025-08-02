import pygame
import sys
import numpy as np
from Game import Game
from GameConfig import *
from DQNNetwork import DQNNetwork

class BattleshipGUI:
    def __init__(self, model_file=None):
        pygame.init()
        self.WHITE = (255, 255, 255)
        self.GRAY = (220, 220, 220)
        self.BLUE = (65, 105, 225) 
        self.ORANGE = (255, 140, 0) 
        self.RED = (220, 20, 60)  
        self.BLACK = (30, 30, 30)
        self.GRID_LINE = (180, 180, 180)
        self.CELL_SIZE = 80
        self.GRID_SIZE = BOARD_WIDTH
        self.MARGIN = 100
        self.GRID_PADDING = 250

        self.WINDOW_WIDTH = self.MARGIN * 2 + self.CELL_SIZE * self.GRID_SIZE * 2 + self.GRID_PADDING
        self.WINDOW_HEIGHT = self.MARGIN * 3 + self.CELL_SIZE * self.GRID_SIZE
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Battleship Battle")
        self.title_font = pygame.font.SysFont('arial', 48, bold=True)
        self.grid_font = pygame.font.SysFont('arial', 36)
        self.cell_font = pygame.font.SysFont('arial', 20)
        self.network = DQNNetwork(BOARD_WIDTH, BOARD_HEIGHT, len(SHIPS))
        if model_file:
            self.network.restoreModel(model_file)
        self.ai_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, network=self.network)  
        self.human_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, network=self.network)  
        self.game_over = False
        self.winner = None
        self.ai_grid_moves = set()  
        self.human_grid_moves = set()  

    def draw_grid_labels(self, x_offset):
        # Draw column labels (numbers)
        for col in range(self.GRID_SIZE):
            label = self.cell_font.render(str(col), True, self.BLACK)
            x = x_offset + col * self.CELL_SIZE + self.CELL_SIZE // 2 + self.MARGIN
            y = self.MARGIN - 30
            self.screen.blit(label, (x - label.get_width() // 2, y))
        # Draw row labels (letters)
        for row in range(self.GRID_SIZE):
            label = self.cell_font.render(str(row), True, self.BLACK)
            x = x_offset + self.MARGIN - 30
            y = row * self.CELL_SIZE + self.CELL_SIZE // 2 + self.MARGIN
            self.screen.blit(label, (x, y - label.get_height() // 2))

    def draw_grid(self, x_offset, grid, is_ai=False):
        # Draw grid title
        title = "AI's Grid" if is_ai else "Your Grid"
        title_surface = self.grid_font.render(title, True, self.BLACK)
        title_x = x_offset + (self.GRID_SIZE * self.CELL_SIZE) // 2 - title_surface.get_width() // 2 + self.MARGIN
        self.screen.blit(title_surface, (title_x, self.MARGIN - 70))
        # Draw grid labels
        self.draw_grid_labels(x_offset)
        # Draw cells
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                x = x_offset + col * self.CELL_SIZE + self.MARGIN
                y = row * self.CELL_SIZE + self.MARGIN
                # Draw cell
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.GRAY, rect)
                pygame.draw.rect(self.screen, self.GRID_LINE, rect, 1)
                # Draw cell state
                cell_state = grid.view_state[row][col]
                if cell_state == 'X':  # Miss
                    pygame.draw.rect(self.screen, self.BLUE, rect)
                    pygame.draw.circle(self.screen, self.WHITE, (x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2), self.CELL_SIZE // 6)
                elif cell_state == 'O':  # Hit
                    pygame.draw.rect(self.screen, self.ORANGE, rect)
                    pygame.draw.line(self.screen, self.WHITE, (x + 20, y + 20), (x + self.CELL_SIZE - 20, y + self.CELL_SIZE - 20), 4)
                    pygame.draw.line(self.screen, self.WHITE, (x + self.CELL_SIZE - 20, y + 20), (x + 20, y + self.CELL_SIZE - 20), 4)
                elif cell_state in ['@', '#']:  # Sunk ship
                    pygame.draw.rect(self.screen, self.RED, rect)
                    pygame.draw.circle(self.screen, self.WHITE, (x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2), self.CELL_SIZE // 3, 3)

    def draw_game_over(self):
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(self.WHITE)
        self.screen.blit(overlay, (0, 0))
        text_color = (34, 139, 34) if self.winner == "Human" else (178, 34, 34)
        font = pygame.font.SysFont('arial', 74, bold=True)
        text = font.render(f"{self.winner} Wins!", True, text_color)
        text_rect = text.get_rect(center=(self.WINDOW_WIDTH / 2, self.WINDOW_HEIGHT / 2))
        shadow = font.render(f"{self.winner} Wins!", True, self.BLACK)
        shadow_rect = shadow.get_rect(center=(self.WINDOW_WIDTH / 2 + 2, self.WINDOW_HEIGHT / 2 + 2))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(text, text_rect)

    def get_cell_from_mouse(self, mouse_pos):
        x, y = mouse_pos
        grid_x_start = self.MARGIN 
        col = int((x - grid_x_start) // self.CELL_SIZE)    
        row = int((y - self.MARGIN) // self.CELL_SIZE)
        if 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE:
            print(f"Mouse click detected at grid position: ({row}, {col})")  
            return row, col
        print(f"Mouse click out of grid bounds: {mouse_pos}") 
        return None

    def is_valid_move(self, row, col):
        move_id = (row, col)
        if not (0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE):
            print(f"Move out of bounds: ({row}, {col})")
            return False
        # Check moves history based on which grid is being targeted
        if move_id in self.ai_grid_moves:
            print(f"Move already made on AI's grid: ({row}, {col})")
            return False
        current_state = self.ai_game.board.view_state[row][col]
        print(f"Move state at ({row}, {col}): {current_state}")
        return current_state not in ['X', 'O', '@', '#']

    def make_human_move(self, row, col):
        location = row * BOARD_WIDTH + col
        result = self.ai_game.takeAMove(location)  # Attack AI's board
        self.ai_grid_moves.add((row, col))  # Add to AI grid moves history
        return result

    def make_ai_move(self):
        result = self.human_game.takeAMove()
        if result and result[1] is not None:
            ai_move = result[1]
            ai_row = ai_move // BOARD_WIDTH
            ai_col = ai_move % BOARD_WIDTH
            self.human_grid_moves.add((ai_row, ai_col))  # Add to human grid moves history

    def handle_game_over(self):
        if self.ai_game.board.checkIfGameFinished():
            self.game_over = True
            self.winner = "Human"
        elif self.human_game.board.checkIfGameFinished():
            self.game_over = True
            self.winner = "AI"

    def play(self):
        while True:
            self.screen.fill(self.WHITE)
            self.draw_grid(0, self.ai_game.board, True)  # AI's grid (left)
            self.draw_grid(self.GRID_SIZE * self.CELL_SIZE + self.GRID_PADDING, self.human_game.board, False)  # Human's grid (right)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    cell = self.get_cell_from_mouse(event.pos)
                    if cell:
                        row, col = cell
                        if self.is_valid_move(row, col):
                            self.make_human_move(row, col)
                            print("AI's grid after move:")
                            for line in self.ai_game.board.view_state:
                                print(line)  
                            self.make_ai_move()
                            print("Human's grid after move:")
                            for line in self.human_game.board.view_state:
                                print(line)  
                            self.handle_game_over()
            if self.game_over:
                self.draw_game_over()
            pygame.display.flip()

if __name__ == "__main__":
    game = BattleshipGUI(r'D:\download\BattleShip\BattleShip\mymodel.keras')
    game.play()