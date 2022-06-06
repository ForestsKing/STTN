import os

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.STTN import STTN
from utils.earlystoping import EarlyStopping
from utils.getdata import get_data


class EXP:
    def __init__(self, his_len=12, pre_len=15, lr=0.001, batch_size=32, epochs=10, patience=3):
        self.his_len = his_len
        self.pre_len = pre_len

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')

        self.modelpath = './checkpoint/STTN.pkl'
        self.scalerpath = './checkpoint/scaler.pkl'

        self._get_data()
        self._get_model()

    def _get_data(self):
        self.scaler = StandardScaler()
        train, valid, test, A = get_data()
        A_wave = A + np.eye(A.shape[0])
        D_wave = np.eye(A.shape[0]) * (np.sum(A_wave, axis=0) ** (-0.5))
        self.adj = np.matmul(np.matmul(D_wave, A_wave), D_wave)

        train = self.scaler.fit_transform(train)
        valid = self.scaler.transform(valid)
        test = self.scaler.transform(test)

        joblib.dump(self.scaler, self.scalerpath)

        trainset = MyDataset(data=train, his_len=self.his_len, pre_len=self.pre_len)
        validset = MyDataset(data=valid, his_len=self.his_len, pre_len=self.pre_len)
        testset = MyDataset(data=test, his_len=self.his_len, pre_len=self.pre_len)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = STTN(adj=self.adj, len_his=self.his_len, len_pred=self.pre_len).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=self.modelpath)
        self.criterion = nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)  # 32 * 12 * 1 * 25
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        return outputs, loss

    def train(self):
        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            for (batch_x, batch_y) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                valid_loss = []
                for (batch_x, batch_y) in tqdm(self.validloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y)
                    valid_loss.append(loss.item())

                test_loss = []
                for (batch_x, batch_y) in tqdm(self.testloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y)
                    test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()
        self.model.load_state_dict(torch.load(self.modelpath))

    def test(self):
        self.model.load_state_dict(torch.load(self.modelpath, map_location=self.device))
        self.model.eval()
        with torch.no_grad():
            trues, preds = [], []
            for (batch_x, batch_y) in tqdm(self.testloader):
                pred, loss = self._process_one_batch(batch_x, batch_y)
                preds.extend(pred.detach().cpu().numpy())
                trues.extend(batch_y.detach().cpu().numpy())

        trues, preds = np.array(trues), np.array(preds)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        print('Test: MSE:{0:.6f}, MAE:{1:.6f}'.format(mse, mae))
