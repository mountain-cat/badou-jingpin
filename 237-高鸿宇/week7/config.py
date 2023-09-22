import torch

class Configures:
    # 数据集文件保存路径
    filepath = 'week7\文本分类练习.csv'

    # 文本最大长度
    max_length = 20

    batch_size = 1024

    hidden_size = 768

    n_classes = 2

    lr = 1e-5

    num_epoch = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 模型权重保存位置
    weight_save_path:str = 'week7'
    weight_to_load:str = 'week7\model.pth'

opt = Configures()