import copy
import random
import numpy as np

class HuntingStrategy:
    def __init__(self, board_width, board_height):
        self.board_width = board_width
        self.board_height = board_height
        self.reset()

    def reset(self):
        self.hits = []  
        self.potential_direction = None 
        self.last_hit = None

    def get_hunt_targets(self, available_moves):
        """Trả về danh sách các vị trí nên tấn công tiếp theo"""
        if not self.hits:
            return None 
        moves = []
        last_hit = self.hits[-1]
        x, y = divmod(last_hit, self.board_width)
        if self.potential_direction == 'horizontal':
            if y > 0:
                moves.append(last_hit - 1)
            if y < self.board_width - 1:
                moves.append(last_hit + 1)
        elif self.potential_direction == 'vertical':
            if x > 0:
                moves.append(last_hit - self.board_width)
            if x < self.board_height - 1:
                moves.append(last_hit + self.board_width)
        else:
            if y > 0:
                moves.append(last_hit - 1)
            if y < self.board_width - 1:
                moves.append(last_hit + 1)
            if x > 0:
                moves.append(last_hit - self.board_width)
            if x < self.board_height - 1:
                moves.append(last_hit + self.board_width)
        valid_moves = [move for move in moves if available_moves[move] == 1]
        return valid_moves if valid_moves else None

    def update(self, move, is_hit, is_ship_sunk):
        """Cập nhật strategy dựa trên kết quả đánh"""
        if is_hit:
            if is_ship_sunk:
                self.reset()
            else:
                self.hits.append(move)
                if len(self.hits) >= 2:
                    x1, y1 = divmod(self.hits[-2], self.board_width)
                    x2, y2 = divmod(move, self.board_width)
                    if x1 == x2:
                        self.potential_direction = 'horizontal'
                    elif y1 == y2:
                        self.potential_direction = 'vertical'

class Board:
    def __init__(self, board_height, board_width, ships):
        self.board_height = board_height
        self.board_width = board_width
        self.state_number = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.true_state = [['-' for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.view_state = [['-' for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.ships = copy.deepcopy(ships)
        for ship in self.ships:
            ship['remaining_length'] = ship['length']
        self.remaining_ships = copy.deepcopy(ships)
        self.available_bomb_locations = np.full(self.board_height * self.board_width, 1, 'float32')
        self.hunting_strategy = HuntingStrategy(board_width, board_height)
        self.randomPlacement()

    def getViewState(self):
        statePrinter = [' ' + ' '.join(str(i) for i in range(self.board_width))]
        for i in range(self.board_height):
            row = str(i) + ' ' + ' '.join(self.view_state[i])
            statePrinter.append(row)
        return statePrinter

    def randomPlacement(self):
        while True:
            ship, available_placements = self.getNextShipAvailablePlacements()
            if ship is None:
                break
            placement = random.choice(available_placements)
            self.placeShip(ship, placement)

    def getNextShipAvailablePlacements(self):
        if not self.remaining_ships:
            return (None, None)
        cur_ship = self.remaining_ships.pop(0)
        ship_length = cur_ship['length']
        available_placement = []
        for i in range(self.board_height):
            for j in range(self.board_width):
                if j + ship_length <= self.board_width and all(self.true_state[i][j + k] == '-' for k in range(ship_length)):
                    available_placement.append({'x': i, 'y': j, 'z': 0})
                if i + ship_length <= self.board_height and all(self.true_state[i + k][j] == '-' for k in range(ship_length)):
                    available_placement.append({'x': i, 'y': j, 'z': 1})
        return (cur_ship, available_placement)

    def placeShip(self, ship, placement):
        x, y, z = placement['x'], placement['y'], placement['z']
        ship_mark = ship['mark']
        ship_length = ship['length']
        for i in range(ship_length):
            if z == 0:
                self.true_state[x][y + i] = ship_mark
            else:
                self.true_state[x + i][y] = ship_mark

    def checkIfGameFinished(self):
        return all(ship['remaining_length'] == 0 for ship in self.ships)

    def getNextAvailableBombLocations(self):
        return copy.deepcopy(self.available_bomb_locations)

    def placeBombAndCheckIfHit(self, location):
        location = int(location)
        self.available_bomb_locations[location] = 0
        x, y = divmod(location, self.board_width)
        is_hit = 0
        is_ship_sunk = False
        if self.true_state[x][y] != '-':
            self.state_number[x][y] = 1
            ship_mark = self.true_state[x][y]
            for ship in self.ships:
                if ship['mark'] == ship_mark:
                    ship['remaining_length'] -= 1
                    is_hit = 1
                    if ship['remaining_length'] == 0:
                        is_ship_sunk = True
                        for i in range(self.board_height):
                            for j in range(self.board_width):
                                if self.true_state[i][j] == ship_mark:
                                    self.view_state[i][j] = ship_mark
            else:
                self.view_state[x][y] = 'O'
        else:
            self.state_number[x][y] = -1
            self.view_state[x][y] = 'X'
        self.hunting_strategy.update(location, is_hit, is_ship_sunk)
        return is_hit

    def getInputDimensions(self):
        input_dimensions = np.array([self.state_number], dtype='float32').flatten()
        for ship in self.ships:
            sink_flag = 0 if ship['remaining_length'] == 0 else 1
            ship_dimension = np.full((self.board_height, self.board_width), sink_flag, dtype='float32').flatten()
            input_dimensions = np.append(input_dimensions, ship_dimension)
        return input_dimensions.reshape(1, -1)

class Game:
    def __init__(self, board_height, board_width, ships, network=None):
        self.network = network
        self.board_height = board_height
        self.board_width = board_width
        self.ships = ships
        self.board = Board(self.board_height, self.board_width, self.ships)

    def resetBoard(self):
        self.board = Board(self.board_height, self.board_width, self.ships)

    def takeAMove(self, next_move=None):
        if self.board.checkIfGameFinished():
            return None, None, None
        input_dimensions = self.board.getInputDimensions()
        available_moves = self.board.getNextAvailableBombLocations()
        if next_move is None:
            hunt_moves = self.board.hunting_strategy.get_hunt_targets(available_moves)
            if hunt_moves:
                next_move = random.choice(hunt_moves)
            else:
                next_move = self.getBestMoveBasedOnModel(input_dimensions, available_moves)
        is_hit = self.board.placeBombAndCheckIfHit(next_move)
        return input_dimensions, next_move, is_hit

    def getBestMoveBasedOnModel(self, input_dimensions, available_moves):
        if self.network and random.random() > self.network.epsilon:
            board_probs = self.network.getBoardProbabilities(input_dimensions)
            board_probs = np.multiply(board_probs, available_moves)
            return np.argmax(board_probs)
        else:
            return self.getRandomMove(available_moves)

    def getRandomMove(self, available_moves):
        available_moves_p = np.copy(available_moves)
        num_available_moves = np.sum(available_moves)
        available_moves_p[available_moves_p != 1] = 0
        available_moves_p[available_moves_p == 1] = 1.0 / num_available_moves
        return np.random.choice(np.arange(len(available_moves)), p=available_moves_p)