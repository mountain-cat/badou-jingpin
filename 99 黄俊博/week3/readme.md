
nlpdemo变成rnn实现

+ 
  _,x=self.rnn(x),x 是预测隐含层结果，之后使用要squeeze去掉多余维度

  x = self.classify(x.squeeze())

  y 真实值也要， ？


