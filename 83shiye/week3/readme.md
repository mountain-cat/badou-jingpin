## Requirement:
Try to train using the RNN model in nlpdemo  

## Solution:
* **1. Add RNN layer:**  
`self.rnn = nn.RNN(vector_dim, vector_dim, 2, batch_first=True)` 

* **2. change forward process:**  
  ```python
  def forward(self, x, y=None):  
        x = self.embedding(x)       
        _, h = self.rnn(x)                                   
        x = self.classify(h[-1])                 
        y_pred = self.activation(x)   
        if y is not None:  
            return self.loss(y_pred, y)  
        else:  
            return y_pred
  ```

* **3. result:**  
  ![](../week2/imgs/week3.png)
