在nlpdemo中使用rnn模型训练
要点：
1、把pooling层改为RNN(self.rnn = nn.RNN(input_size=vector_dim, hidden_size=vector_dim, num_layers=1, batch_first=True))
2、