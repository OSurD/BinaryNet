from BinaryNet.binary_layers.XNORFromRepository import BinOp
from BinaryNet.utils.ClassificationTrainer import ClassificationTrainer

import torch
from sklearn.metrics import f1_score
import tqdm

class BinOpClassificationTrainer(ClassificationTrainer):
    def __init__(self, train, test, loss_f = torch.nn.CrossEntropyLoss):
        super().__init__(train, test, loss_f)

    def _test_step(self, Net, test, bin_op):
        loss_func = self.loss_f()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Net.eval()
        with torch.no_grad():
            bin_op.binarization() #< use bin_op
            series_test, labels_test = test
            pred = Net(series_test.to(device))
            loss = loss_func(pred, labels_test.to(device))
            y_p = self._soft_argmax(pred)
            f1 = f1_score(labels_test, y_p.to("cpu"), average='macro')
            bin_op.restore() #< use bin_op
            return loss, f1

    def fit(self, 
            num_epochs, model, 
            bs_train = 128, 
            bs_test = 256, 
            optim = torch.optim.Adam, 
            lr = 0.001, 
            save_path = None):
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

        bin_op = BinOp(model) #< use bin_op

        for epoch in range(num_epochs):
            for i, (series, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                Net.train()
                bin_op.binarization() #< use bin_op
                outputs = Net(series.to(device))
                loss = loss_func(outputs, labels.to(device))
                self.loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                bin_op.restore() #< use bin_op
                bin_op.updateBinaryGradWeight() #< use bin_op

                optimizer.step()

                with torch.no_grad():
                    y_p = self._soft_argmax(outputs)
                    self.f1_list.append(f1_score(labels, y_p.to("cpu"), average='macro'))
                    
                    test_loss = self._test_step(Net, test_data, bin_op)
                    self.loss_test_list.append(test_loss[0].item())
                    self.f1_test_list.append(test_loss[1])

            print(f'epoch - {epoch}: loss = {self.loss_test_list[-1]}, f1 = {self.f1_test_list[-1]}')

        if save_path:
            self._checkpoint(Net, save_path, f"epoch_{epoch}")
        self.current_net = Net
        return Net