#đã xong mục này
import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any
from fractions import Fraction
import random
from collections import deque
import sys
import os
import json

# tạo rắn và các lớp

from misc import *
from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name



class Vision(object):
    __slots__ = ('dist_to_wall', 'dist_to_apple', 'dist_to_self')
    def __init__(self,
                 dist_to_wall: Union[float, int],
                 dist_to_apple: Union[float, int],
                 dist_to_self: Union[float, int]
                 ):
        self.dist_to_wall = float(dist_to_wall)
        self.dist_to_apple = float(dist_to_apple)
        self.dist_to_self = float(dist_to_self)

class DrawableVision(object):
    __slots__ = ('wall_location', 'apple_location', 'self_location')
    def __init__(self,
                wall_location: Point,
                apple_location: Optional[Point] = None,
                self_location: Optional[Point] = None,
                ):
        self.wall_location = wall_location
        self.apple_location = apple_location
        self.self_location = self_location


class Snake(Individual):
    def __init__(self, board_size: Tuple[int, int],
                 chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                 start_pos: Optional[Point] = None, 
                 apple_seed: Optional[int] = None,
                 initial_velocity: Optional[str] = None,
                 starting_direction: Optional[str] = None,
                 hidden_layer_architecture: Optional[List[int]] = [1123125, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu', # các hàm
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 lifespan: Optional[Union[int, float]] = np.inf,
                 apple_and_self_vision: Optional[str] = 'binary'
                 ):

        self.lifespan = lifespan
        self.apple_and_self_vision = apple_and_self_vision.lower()
        self.score = 0  # Number of apples snake gets
        self._fitness = 0  # Overall fitness
        self._frames = 0  # Number of frames that the snake has been alive
        self._frames_since_last_apple = 0
        self.possible_directions = ('u', 'd', 'l', 'r')

        self.board_size = board_size
        self.hidden_layer_architecture = hidden_layer_architecture

        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if not start_pos:
            #@TODO: dưới
            # x = random.randint(10, self.board_size[0] - 9)
            # y = random.randint(10, self.board_size[1] - 9)
            x = random.randint(2, self.board_size[0] - 3)
            y = random.randint(2, self.board_size[1] - 3)

            start_pos = Point(x, y)
        self.start_pos = start_pos

        self._vision_type = VISION_16                                                   ######### Vẽ tầm nhìn ở đây nè !!!!
        self._vision: List[Vision] = [None] * len(self._vision_type)
        # Cái này chỉ được sử dụng để có thể vẽ và không thực sự được sử dụng trong NN
        self._drawable_vision: List[DrawableVision] = [None] * len(self._vision_type)

        # Thiết lập kiến trúc mạng
        # Mỗi "Tầm nhìn" có 3 khoảng cách mà nó theo dõi: tường, táo và bản thân nó
        # cũng có hướng được mã hóa nóng và hướng đuôi được mã hóa nóng,
        # mỗi trong số đó có 4 khả năng.
        num_inputs = len(self._vision_type) * 3 + 4 + 4 #@TODO: Add one-hot back in
        self.vision_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Đầu ra
        self.network_architecture.extend(self.hidden_layer_architecture)  # Lớp ẩn
        self.network_architecture.append(4)                               # 4 đầu ra, ['u', 'd', 'l', 'r']
        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
        )

        # Nếu nhiễm sắc thể được thiết lập, hãy lấy nó
        if chromosome:
            # self._chromosome = chromosome
            self.network.params = chromosome
            # self.decode_chromosome()
        else:
            # self._chromosome = {}
            # self.encode_chromosome()
            pass
            

        # Tạo mồi tiếp theo
        if apple_seed is None:
            apple_seed = np.random.randint(-1000000000, 1000000000)#thức ăn
        self.apple_seed = apple_seed  # Chỉ cần để lưu / tải phát lại
        self.rand_apple = random.Random(self.apple_seed)

        self.apple_location = None
        if starting_direction:
            starting_direction = starting_direction[0].lower()
        else:
            starting_direction = self.possible_directions[random.randint(0, 3)]

        self.starting_direction = starting_direction  # Chỉ cần để lưu / tải phát lại
        self.init_snake(self.starting_direction)
        self.initial_velocity = initial_velocity
        self.init_velocity(self.starting_direction, self.initial_velocity)
        self.generate_apple()

    @property
    def fitness(self):
        return self._fitness
    
    def calculate_fitness(self):
        # Cung cấp cho quần thể tối thiểu tích cực để lựa chọn thế hệ sau
        self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)**1.3) * (self.score**1.2))
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)) * (self.score))
        self._fitness = max(self._fitness, .1)

    @property
    def chromosome(self):
        # return self._chromosome
        pass

    def encode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Encode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     self._chromosome['W' + l] = self.network.params['W' + l].flatten()
        #     self._chromosome['b' + l] = self.network.params['b' + l].flatten()
        pass

    def decode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Decode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     w_shape = (self.network_architecture[layer], self.network_architecture[layer-1])
        #     b_shape = (self.network_architecture[layer], 1)
        #     self.network.params['W' + l] = self._chromosome['W' + l].reshape(w_shape)
        #     self.network.params['b' + l] = self._chromosome['b' + l].reshape(b_shape)
        pass

    def look(self):
        # Look all around
        for i, slope in enumerate(self._vision_type):
            vision, drawable_vision = self.look_in_direction(slope)
            self._vision[i] = vision
            self._drawable_vision[i] = drawable_vision
        
        # Update the input array
        self._vision_as_input_array()


    def look_in_direction(self, slope: Slope) -> Tuple[Vision, DrawableVision]:
        dist_to_wall = None
        dist_to_apple = np.inf
        dist_to_self = np.inf

        wall_location = None
        apple_location = None
        self_location = None

        position = self.snake_array[0].copy()
        distance = 1.0
        total_distance = 0.0

        # Can't start by looking at yourself
        position.x += slope.run
        position.y += slope.rise
        total_distance += distance
        body_found = False  # Chỉ cần tìm sự xuất hiện đầu tiên vì nó là lần gần nhất
        food_found = False  # Mặc dù chỉ có một thức ăn, hãy ngừng tìm kiếm khi tìm thấy nó

        # Tiếp tục đi cho đến khi vị trí nằm ngoài giới hạn
        while self._within_wall(position):
            if not body_found and self._is_body_location(position):
                dist_to_self = total_distance
                self_location = position.copy()
                body_found = True
            if not food_found and self._is_apple_location(position):
                dist_to_apple = total_distance
                apple_location = position.copy()
                food_found = True

            wall_location = position
            position.x += slope.run
            position.y += slope.rise
            total_distance += distance
        assert(total_distance != 0.0)


        # @TODO: Có thể cần điều chỉnh tử số trong trường hợp VISION_16 vì kích thước bước không phải lúc nào cũng nằm trên ô
        dist_to_wall = 1.0 / total_distance

        if self.apple_and_self_vision == 'binary':
            dist_to_apple = 1.0 if dist_to_apple != np.inf else 0.0
            dist_to_self = 1.0 if dist_to_self != np.inf else 0.0

        elif self.apple_and_self_vision == 'distance':
            dist_to_apple = 1.0 / dist_to_apple
            dist_to_self = 1.0 / dist_to_self

        vision = Vision(dist_to_wall, dist_to_apple, dist_to_self)
        drawable_vision = DrawableVision(wall_location, apple_location, self_location)
        return (vision, drawable_vision)

    def _vision_as_input_array(self) -> None:
        # Split _vision into np array where rows [0-2] are _vision[0].dist_to_wall, _vision[0].dist_to_apple, _vision[0].dist_to_self,
        # rows [3-5] are _vision[1].dist_to_wall, _vision[1].dist_to_apple, _vision[1].dist_to_self
        for va_index, v_index in zip(range(0, len(self._vision) * 3, 3), range(len(self._vision))):
            vision = self._vision[v_index]
            self.vision_as_array[va_index, 0]     = vision.dist_to_wall
            self.vision_as_array[va_index + 1, 0] = vision.dist_to_apple
            self.vision_as_array[va_index + 2, 0] = vision.dist_to_self

        i = len(self._vision) * 3  # Start at the end

        direction = self.direction[0].lower()
        # One-hot encode direction
        direction_one_hot = np.zeros((len(self.possible_directions), 1))
        direction_one_hot[self.possible_directions.index(direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = direction_one_hot

        i += len(self.possible_directions)

        # One-hot tail direction
        tail_direction_one_hot = np.zeros((len(self.possible_directions), 1))
        tail_direction_one_hot[self.possible_directions.index(self.tail_direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = tail_direction_one_hot

    def _within_wall(self, position: Point) -> bool:
        return position.x >= 0 and position.y >= 0 and \
               position.x < self.board_size[0] and \
               position.y < self.board_size[1]

    def generate_apple(self) -> None:
        width = self.board_size[0]
        height = self.board_size[1]
        # Tìm tất cả các điểm có thể mà con rắn hiện không ở đó
        possibilities = [divmod(i, height) for i in range(width * height) if divmod(i, height) not in self._body_locations]
        if possibilities:
            loc = self.rand_apple.choice(possibilities)
            self.apple_location = Point(loc[0], loc[1])
        else:
            print('Bạn thắng!')
            pass

    def init_snake(self, starting_direction: str) -> None:
        """
        Khởi tạo con rắn.
start_direction: ('u', 'd', 'l', 'r')
hướng mà con rắn nên bắt đầu phải đối mặt. Dù là hướng nào, cái đầu
của con rắn sẽ bắt đầu chỉ theo cách đó.
        """        
        head = self.start_pos
        # Body is below
        if starting_direction == 'u':
            snake = [head, Point(head.x, head.y + 1), Point(head.x, head.y + 2)]
        # Body is above
        elif starting_direction == 'd':
            snake = [head, Point(head.x, head.y - 1), Point(head.x, head.y - 2)]
        # Body is to the right
        elif starting_direction == 'l':
            snake = [head, Point(head.x + 1, head.y), Point(head.x + 2, head.y)]
        # Body is to the left
        elif starting_direction == 'r':
            snake = [head, Point(head.x - 1, head.y), Point(head.x - 2, head.y)]

        self.snake_array = deque(snake)
        self._body_locations = set(snake)
        self.is_alive = True

    def update(self):
        if self.is_alive:
            self._frames += 1
            self.look()
            self.network.feed_forward(self.vision_as_array)
            self.direction = self.possible_directions[np.argmax(self.network.out)]
            return True
        else:
            return False

    def move(self) -> bool:
        if not self.is_alive:
            return False

        direction = self.direction[0].lower()
        # Is the direction valid?
        if direction not in self.possible_directions:
            return False
        
        # Tìm vị trí tiếp theo
        # tail = self.snake_array.pop()  # Pop tail vì kỹ thuật chúng ta có thể di chuyển đến đuôi
        head = self.snake_array[0]
        if direction == 'u':
            next_pos = Point(head.x, head.y - 1)
        elif direction == 'd':
            next_pos = Point(head.x, head.y + 1)
        elif direction == 'r':
            next_pos = Point(head.x + 1, head.y)
        elif direction == 'l':
            next_pos = Point(head.x - 1, head.y)

        # Là vị trí tiếp theo chúng ta muốn di chuyển hợp lệ?
        if self._is_valid(next_pos):
            # Tail
            if next_pos == self.snake_array[-1]:
                # Pop tail and add next_pos (same as tail) to front
                # No need to remove tail from _body_locations since it will go back in anyway
                self.snake_array.pop()
                self.snake_array.appendleft(next_pos) 
            # ăn táo
            elif next_pos == self.apple_location:
                self.score += 1
                self._frames_since_last_apple = 0
                # Move head
                self.snake_array.appendleft(next_pos)
                self._body_locations.update({next_pos})
                # Đừng bỏ đuôi từ khi con rắn lớn lên
                self.generate_apple()
            # Normal movement
            else:
                # Move head
                self.snake_array.appendleft(next_pos)
                self._body_locations.update({next_pos})
                # Remove tail
                tail = self.snake_array.pop()
                self._body_locations.symmetric_difference_update({tail})

            # Chỉ ra hướng di chuyển của đuôi
            p2 = self.snake_array[-2]
            p1 = self.snake_array[-1]
            diff = p2 - p1
            if diff.x < 0:
                self.tail_direction = 'l'
            elif diff.x > 0:
                self.tail_direction = 'r'
            elif diff.y > 0:
                self.tail_direction = 'd'
            elif diff.y < 0:
                self.tail_direction = 'u'

            self._frames_since_last_apple += 1
            #@NOTE: Nếu ta có lưới có kích thước khác nhau, có thể muốn thay đổi điều này
            if self._frames_since_last_apple > 100:
                self.is_alive = False
                return False

            return True
        else:
            self.is_alive = False
            return False

    def _is_apple_location(self, position: Point) -> bool:
        return position == self.apple_location

    def _is_body_location(self, position: Point) -> bool:
        return position in self._body_locations

    def _is_valid(self, position: Point) -> bool:
        """
        Xác định xem một vị trí nhất định có hợp lệ không.
         Trả về True nếu vị trí nằm trên bảng và không giao nhau với con rắn.
         Trả về Sai nếu không
        """
        if (position.x < 0) or (position.x > self.board_size[0] - 1):
            return False
        if (position.y < 0) or (position.y > self.board_size[1] - 1):
            return False

        if position == self.snake_array[-1]:
            return True
        # Nếu vị trí là một vị trí cơ thể, không hợp lệ.
        # @Note: _body_locations sẽ chứa đuôi, vì vậy cần kiểm tra đuôi trước
        elif position in self._body_locations:
            return False
        # Otherwise you good
        else:
            return True

    def init_velocity(self, starting_direction, initial_velocity: Optional[str] = None) -> None:
        if initial_velocity:
            self.direction = initial_velocity[0].lower()
        # Dù cách khởi động là gì
        else:
            self.direction = starting_direction

        # Đuôi bắt đầu di chuyển cùng hướng
        self.tail_direction = self.direction

def save_snake(population_folder: str, individual_name: str, snake: Snake, settings: Dict[str, Any]) -> None:
    # Tạo thư mục dân số nếu nó không tồn tại
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # Lưu cài đặt
    if 'settings.json' not in os.listdir(population_folder):
        f = os.path.join(population_folder, 'settings.json')
        with open(f, 'w', encoding='utf-8') as out:
            json.dump(settings, out, sort_keys=True, indent=4)

    # Tạo thư mục cho cá nhân
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    # Lưu một số thông tin về hàm tạo để phát lại
    # @Note: Không cần lưu nhiễm sắc thể vì đó được lưu dưới dạng .npy
    # @NOTE: Không cần lưu board_size hoặc hidden_layer_arch architecture
    # vì chúng được lấy từ cài đặt














    constructor = {}
    constructor['start_pos'] = snake.start_pos.to_dict()
    constructor['apple_seed'] = snake.apple_seed
    constructor['initial_velocity'] = snake.initial_velocity
    constructor['starting_direction'] = snake.starting_direction
    snake_constructor_file = os.path.join(individual_dir, 'constructor_params.json')

    # Save
    with open(snake_constructor_file, 'w', encoding='utf-8') as out:
        json.dump(constructor, out, sort_keys=True, indent=4)

    L = len(snake.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = snake.network.params[w_name]
        bias = snake.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)

def load_snake(population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Snake:
    if not settings:
        f = os.path.join(population_folder, 'settings.json')
        if not os.path.exists(f):
            raise Exception("settings needs to be passed as an argument if 'settings.json' does not exist under population folder")
        
        with open(f, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    elif isinstance(settings, dict):
        settings = settings

    elif isinstance(settings, str):
        filepath = settings
        with open(filepath, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    params = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            params[param] = np.load(os.path.join(population_folder, individual_name, fname))
        else:
            continue

    # Load constructor params for the specific snake
    constructor_params = {}
    snake_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
    with open(snake_constructor_file, 'r', encoding='utf-8') as fp:
        constructor_params = json.load(fp)

    snake = Snake(settings['board_size'], chromosome=params, 
                  start_pos=Point.from_dict(constructor_params['start_pos']),
                  apple_seed=constructor_params['apple_seed'],
                  initial_velocity=constructor_params['initial_velocity'],
                  starting_direction=constructor_params['starting_direction'],
                  hidden_layer_architecture=settings['hidden_network_architecture'],
                  hidden_activation=settings['hidden_layer_activation'],
                  output_activation=settings['output_layer_activation'],
                  lifespan=settings['lifespan'],
                  apple_and_self_vision=settings['apple_and_self_vision']
                  )
    return snake