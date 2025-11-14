import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import os

class ClassificationTrainer():
    def __init__(self, train, test, loss_f = torch.nn.CrossEntropyLoss):
        self.train_ds = train
        self.test_ds = test
        self.loss_f = loss_f

    def _soft_argmax(self, prediction):
        return torch.nn.Softmax(dim = 1)(prediction).argmax(1)

    def _test_step(self, Net, test):
        loss_func = self.loss_f()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Net.eval()
        with torch.no_grad():
            series_test, labels_test = test
            pred = Net(series_test.to(device))
            loss = loss_func(pred, labels_test.to(device))
            y_p = self._soft_argmax(pred)
            f1 = f1_score(labels_test, y_p.to("cpu"), average='macro')
            return loss, f1

    def _checkpoint(self, model, path, name):
        torch.save(model.state_dict(), os.path.join(path, f"{name}.pth"))

    def fit(self, 
            num_epochs, model, 
            bs_train = 128, 
            bs_test = 256, 
            optim = torch.optim.Adam, 
            lr = 0.001, 
            save_path = None,
            save_interim_models = False):
        """
        num_epochs - число эпох
        model - модель для обучения
        bs_train - batch size
        optim - оптимизатор
        lr - learning rate
        save_path - путь для сохранения весов
        *args, **kwargs - аргументы модели
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        Net = model.to(device)
        loss_func = self.loss_f()
        optimizer = optim(Net.parameters(), lr=lr)
        train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=bs_train, shuffle=True, num_workers=4, persistent_workers=True, multiprocessing_context="spawn")
        test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=bs_test, shuffle=True)

        test_data = next(iter(test_loader))

        self.loss_list = []
        self.loss_test_list = []
        self.f1_list = []
        self.f1_test_list = []

        for epoch in range(num_epochs):
            for i, (series, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                Net.train()
                outputs = Net(series.to(device))
                loss = loss_func(outputs, labels.to(device))
                self.loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    y_p = self._soft_argmax(outputs)
                    self.f1_list.append(f1_score(labels, y_p.to("cpu"), average='macro'))
                    
                    test_loss = self._test_step(Net, test_data)
                    self.loss_test_list.append(test_loss[0].item())
                    self.f1_test_list.append(test_loss[1])

            print(f'epoch - {epoch}: loss = {self.loss_test_list[-1]}, f1 = {self.f1_test_list[-1]}')
            if save_path and save_interim_models:
                self._checkpoint(Net, save_path, f"epoch_{epoch}")

        if save_path:
            self._checkpoint(Net, save_path, f"epoch_{epoch}")
        self.current_net = Net
        return Net

    def visualize(self, filename = None):
        fig, ax = plt.subplots(1, 2, figsize = (15, 5))
        ax[0].plot(np.arange(len(self.loss_list)), self.loss_list, label = 'loss_train')
        ax[0].plot(np.arange(len(self.loss_test_list)), self.loss_test_list, label = 'loss_test')
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[1].plot(np.arange(len(self.loss_list)), self.f1_list, label = 'f1_train')
        ax[1].plot(np.arange(len(self.loss_list)), self.f1_test_list, label = 'f1_test')
        ax[1].legend()

        if filename is not None:
            plt.savefig(filename)
        plt.show()