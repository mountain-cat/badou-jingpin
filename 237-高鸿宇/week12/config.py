class Config:

    # label路径
    schema_path = 'badouai/week12/ner_data/schema.json'
    
    # 训练集路径
    train_data_path = 'badouai/week12/ner_data/train.txt'

    # 验证集路径
    valid_data_path = 'badouai/week12/ner_data/test.txt'

    # 模型保存路径
    model_path = 'badouai/week12'

    # 句子最大长度
    max_length = 150

    # 隐藏层维度
    hidden_size = 768

    # 训练轮数
    epoch = 20

    # batch size
    batch_size = 64

    # 学习率
    lr = 1e-3

    # 类别数量
    n_class = 9

    # device
    device = 'cuda:0'

opt = Config()