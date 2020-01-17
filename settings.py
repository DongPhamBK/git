#đã xong mục này
import numpy as np

# cài đặt các thông số
settings = {
    'board_size':                  (10, 10), # Khung trò chơi của rắn

    #### Công cụ liên quan đến mạng thần kinh ####

# Kích hoạt lớp ẩn dành riêng cho các lớp ẩn, tức là tất cả các lớp ngoại trừ đầu vào và đầu ra
    'hidden_layer_activation':     'relu',     # chọn  [relu, sigmoid, tanh, linear, leaky_relu]
# Kích hoạt lớp đầu ra là dành riêng cho lớp đầu ra
    'output_layer_activation':     'sigmoid',  # chọn [relu, sigmoid, tanh, linear, leaky_relu]
# Kiến trúc mạng ẩn mô tả số lượng nút trong mỗi lớp ẩn
    'hidden_network_architecture': [20, 12],   # Một danh sách chứa số nút trong mỗi lớp ẩn
# Số hướng mà con rắn có thể "nhìn thấy" trong trò chơi
    'vision_type':                 16,          # chọn [4, 8, 16]

    #### GA_Settings ####

    ## Đột biến ##

    # Tỷ lệ đột biến là xác suất một gen nhất định trong nhiễm sắc thể sẽ đột biến ngẫu nhiên
    'mutation_rate':               0.05,       # giữa [0.00, 1.00)
    # Nếu loại tỷ lệ đột biến là tĩnh, thì tỷ lệ đột biến sẽ luôn là 'mutation_rate',
    # nếu không, nó sẽ phân rã, nó sẽ giảm khi số lượng thế hệ tăng lên
    'mutation_rate_type':          'static',   # chọn [static, decaying]
    # Xác suất xảy ra nếu đột biến xảy ra, đó là gaussian
    'probability_gaussian':        1.0,        # giữa [0.00, 1.00]
     # Xác suất xảy ra nếu đột biến xảy ra, đó là sự ngẫu nhiên
    'probability_random_uniform':  0.0,        # giữa [0.00, 1.00]

    ## Crossover ##
# eta liên quan đến SBX. Các giá trị lớn hơn tạo ra sự phân phối gần hơn với cha mẹ trong khi các giá trị nhỏ hơn liên quan đến chúng nhiều hơn.
# Chỉ được sử dụng nếu xác suất_SBX> 0,00
    'SBX_eta':                     100,
# Xác suất xảy ra khi sự giao nhau xảy ra, nó được mô phỏng chéo
    'probability_SBX':             0.5,
# Loại SPBX cần xem xét. Nếu nó là 'r' thì nó làm phẳng một mảng 2D theo thứ tự chính của hàng.
# Nếu SPBX_type là 'c' thì nó sẽ làm phẳng một mảng 2 chiều theo thứ tự chính của cột.
    'SPBX_type':                   'r',        # chọn r hoăc c
# Xác suất xảy ra khi sự giao nhau xảy ra, đó là sự giao nhau nhị phân một điểm
    'probability_SPBX':            0.5,
# Loại lựa chọn chéo xác định cách chúng ta chọn các cá nhân cho chéo
    'crossover_selection_type':    'roulette_wheel',

    ## Lựa chọn phối ##

# Số lượng cha mẹ sẽ được sử dụng để tái sinh sản
    'num_parents':                 500,
# Số con sẽ được tạo.  Giữ cho num_offspring> = num_parents
    'num_offspring':               500,
# Loại lựa chọn để sử dụng cho thế hệ tiếp theo.
# Nếu select_type == 'plus':
# Sau đó, num_parents hàng đầu sẽ được chọn từ (num_offspring + num_parents)
# Nếu select_type == 'comma':
# Sau đó, num_parents hàng đầu sẽ được chọn từ (num_offspring)
# @Note: nếu tuổi thọ của cá nhân là 1, thì không thể chọn nó cho thế hệ tiếp theo
# Nếu không thể chọn đủ số lượng cá nhân cho thế hệ tiếp theo, những người ngẫu nhiên mới sẽ thay thế họ.
# @Note: Nếu select_type == 'comma' thì tuổi thọ bị bỏ qua.
# Điều này tương đương với tuổi thọ = 1 trong trường hợp này vì cha mẹ không bao giờ chuyển sang thế hệ mới.
    'selection_type':              'plus',     # có thể là ['plus', 'comma']


# Một cá nhân được phép ở trong quần thể bao lâu trước khi chết.
# Điều này có thể hữu ích để cho phép thăm dò. Hãy tưởng tượng một cá nhân hàng đầu liên tục được lựa chọn cho chéo.
# Với tuổi thọ, sau một số thế hệ, cá nhân sẽ không được chọn tham gia trong tương lai
# các thế hệ. Điều này có thể cho phép các cá nhân khác có áp lực chọn lọc cao hơn so với trước đây.
# @Note điều này chỉ quan trọng nếu 'selecton_type' == 'plus'. Nếu 'select_type' == 'dấu phẩy', thì 'tuổi thọ' hoàn toàn bị bỏ qua.
    'lifespan':                    np.inf,
# Tùy chọn là bất kỳ số nguyên dương hoặc np.inf (được nhập như thể đó là một số, tức là không có dấu ngoặc kép để biến nó thành một chuỗi)
# Loại thị giác mà con rắn có khi nhìn thấy chính nó hoặc quả táo.
# Nếu tầm nhìn là nhị phân, thì đầu vào vào Mạng thần kinh là 1 (có thể thấy) hoặc 0 (không thể thấy).
# Nếu tầm nhìn là khoảng cách, thì đầu vào vào Mạng thần kinh là 1.0 / khoảng cách.
# 1.0 / khoảng cách được sử dụng để giữ giá trị giới hạn ở mức 1.0 là tối đa.
    'apple_and_self_vision':       'distance'    # có thể là ['binary', 'distance']

}