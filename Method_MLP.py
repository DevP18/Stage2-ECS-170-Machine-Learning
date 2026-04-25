from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Method_MLP(method, nn.Module):

    data = None
    max_epoch = 100
    learning_rate = 0.001

    def __init__(self, mName, mDescription, input_dim, num_classes):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss_list = []

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, X, y):

        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        evaluator = Evaluate_Accuracy()

        for epoch in range(self.max_epoch):

            y_pred = self.forward(X)
            loss = self.loss_fn(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_list.append(loss.item())

            if epoch % 10 == 0:
                evaluator.data = {
                    'true_y': y,
                    'pred_y': y_pred.argmax(dim=1)
                }
                print(f"Epoch {epoch} | Loss {loss.item():.4f} | Acc {evaluator.evaluate():.4f}")

        plt.plot(self.loss_list)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def test(self, X):
        X = torch.FloatTensor(np.array(X))
        y_pred = self.forward(X)
        return y_pred.argmax(dim=1).numpy()

    def run(self):
        print("method running...")
        print("training...")

        self.train_model(
            self.data['train']['X'],
            self.data['train']['y']
        )

        print("testing...")

        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y']
        }