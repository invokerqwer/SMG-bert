import pandas as pd
from easydict import EasyDict


from utils import get_logger, calculate_recovery
import os
from sklearn.metrics import r2_score, roc_auc_score
import numpy as np
import torch
from tqdm import tqdm
from datasets import get_dataloader, get_stratify_dataloader


class Downstream_Trainer():
    def __init__(self,  config, model,):
        self.config = config


        self.model = model.to(self.config.device)

        res_dir = self.config.res_dir
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)


        base_name = os.path.basename(self.config.dataset.dataset_path).split(".")[0]


        self.config.model_save_path = os.path.join(res_dir,base_name+".pth")
        self.config.log_path = os.path.join(res_dir,base_name+".log")
        self.config.res_path = os.path.join(res_dir,base_name+".csv")


        self.logger = get_logger(config.log_path)
        self.logger.info(f"model_save_path:{self.config.model_save_path},log_path:{self.config.log_path},res_path:{self.config.res_path}")
        self.logger.info(f"config: {config}")
        if config.dataset.stratify:
            self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_stratify_dataloader(config,self.logger)
        else:
            print("获取随机采样")
            self.train_dataloader,self.valid_dataloader,self.test_dataloader = get_dataloader(config,self.logger)

        if self.config.task_type=="classify":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.MSELoss(reduction="mean")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.AdamW.lr, eps=self.config.train.AdamW.eps,
                                           betas=(self.config.train.AdamW.beta1, self.config.train.AdamW.beta2))
        if config.pretrain_model_path:
            self.logger.info(f"loading pretrain model at {config.pretrain_model_path}")
            self.load_model()

        self.train_losses = []
        self.valid_losses = []
        self.val_metric_list = []
        self.train_metric_list = []

        self.best_metric = None
        self.best_epoch = 0


    def train_iterations(self):
        self.model.train()
        losses = []
        label_list = []
        output_list = []
        for i, batch_data in enumerate(self.train_dataloader):

            adjoin_matrix = batch_data['adjoin_matrix'].to(self.config.device)
            mol_ids = batch_data["mol_ids"].to(self.config.device)
            nmr = batch_data["nmr"].to(self.config.device)
            labels = batch_data["labels"].to(self.config.device)
            output = self.model(mol_ids, nmr, adjoin_matrix)
            loss = self.loss_fn(output.squeeze(-1).float() , labels.float())  # 计算损失

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            label_list += labels.cpu().numpy().tolist()
            output_list += output.detach().cpu().numpy().tolist()
        self.train_losses += losses
        trn_loss = np.array(losses).mean()
        if self.config.task_type == "classify":
            metric = roc_auc_score(label_list, output_list)
        else:
            metric = r2_score(label_list, output_list)
        return EasyDict({"loss": trn_loss, "metric": metric})

    def valid_iterations(self,mode="valid"):

        self.model.eval()
        losses = []
        label_list = []
        output_list = []
        if mode == "valid":
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                adjoin_matrix = batch_data['adjoin_matrix'].to(self.config.device)
                mol_ids = batch_data["mol_ids"].to(self.config.device)
                nmr = batch_data["nmr"].to(self.config.device)
                labels = batch_data["labels"].to(self.config.device)
                output = self.model(mol_ids, nmr, adjoin_matrix)
                loss = self.loss_fn(output.squeeze(-1).float() , labels.float())  # 计算损失
                losses.append(loss.item())
                label_list += labels.cpu().numpy().tolist()
                output_list += output.detach().cpu().numpy().tolist()
        self.valid_losses += losses
        val_loss = np.array(losses).mean()
        if self.config.task_type == "classify":
            metric = roc_auc_score(label_list, output_list)
        else:
            metric = r2_score(label_list, output_list)

        return EasyDict({"loss": val_loss, "metric": metric})



    def save_model(self):
        torch.save(self.model.state_dict(), self.config.model_save_path)


    def load_model(self):

        d = torch.load(self.config.pretrain_model_path, map_location=self.config.device)
        new_dict = {key: val for key, val in d.items() if not key.startswith("downstream_predict_block")}
        self.model.load_state_dict(new_dict, strict=False)

    def train(self,epochs=50):
        if not os.path.exists(os.path.dirname(self.config.model_save_path)):
            os.makedirs(os.path.dirname(self.config.model_save_path))

        for epoch in tqdm(range(1,epochs+1),leave=False):
            train_metric = self.train_iterations()
            val_metric = self.valid_iterations()
            self.train_metric_list.append(train_metric)
            self.val_metric_list.append(val_metric)


            self.logger.info(f'epoch: {epoch}, train_metric: {train_metric}, val_metric: {val_metric}')
            # print(f'epoch: {epoch}, train_metric: {train_metric}, val_metric: {val_metric}')

            if val_metric.metric == np.array([val_metric.metric for val_metric in self.val_metric_list]).max():
                self.save_model()
                test_metric = self.valid_iterations(mode="test")
                self.best_epoch = epoch
                self.best_metric = test_metric.metric
                self.logger.info(f"save model at {self.config.model_save_path}")
                self.logger.info(f"best metric {test_metric}")
                # print(f"save model at {self.config.model_save_path}")
                # print(f"best metric {test_metric}")
        best_metric = self.best_metric
        best_epoch = self.best_epoch
        self.logger.info(f"best_epoch:{best_epoch},best_metric:{best_metric}")
        res = pd.DataFrame({"best_epoch": [best_epoch], "best_metric": [best_metric]})
        res.to_csv(self.config.res_path)

class Pretrain_Trainer():
    def __init__(self,  config, model):
        self.config = config
        self.device = config.device
        self.model = model.to(self.config.device)
        res_dir = self.config.res_dir
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        base_name = os.path.basename(self.config.dataset.dataset_path).split(".")[0]
        self.config.log_path = os.path.join(res_dir,base_name+".log")
        self.config.res_path = os.path.join(res_dir,base_name+".csv")
        self.logger = get_logger(config.log_path)
        self.logger.info(f"model_save_dir:{self.config.res_dir},log_path:{self.config.log_path},res_path:{self.config.res_path}")

        self.logger.info(f"config: {config}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.AdamW.lr,
                                           eps=self.config.train.AdamW.eps,
                                           betas=(self.config.train.AdamW.beta1, self.config.train.AdamW.beta2))
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloader(config, self.logger,pretrain=True)

        self.train_losses_atom = []
        self.train_losses_nmr = []
        self.train_losses = []
        self.train_recovery = []

        self.valid_losses_atom = []
        self.valid_losses_nmr = []
        self.valid_losses = []
        self.valid_recovery = []


        self.test_losses_atom = []
        self.test_losses_nmr = []
        self.test_losses = []
        self.test_recovery = []

        self.step = 0
        self.epoch = 0

    def train(self,epochs=50):
        train_metric_list = []
        valid_metric_list = []
        for epoch in range(1,epochs+1):
            train_metric = self.train_iterations()
            valid_metric = self.valid_test_iterations()

            self.logger.info(f"epoch:{epoch},train_metric:{train_metric},valid_metric:{valid_metric}")

            train_metric_list.append(train_metric)
            valid_metric_list.append(valid_metric)
            self.save_model(os.path.join(self.config.res_dir,str(epoch)+".pth"))

        return train_metric_list,valid_metric_list
    def compute_loss_atom(self,atom_pred,atom_labels,masked_flag):
        atom_pred = atom_pred[masked_flag==1]
        atom_labels = atom_labels[masked_flag==1]
        loss_atom = torch.nn.functional.cross_entropy(atom_pred, atom_labels)
        return loss_atom


    def train_iterations(self):
        self.model.train()
        cur_losses = []
        for tra_step, batch_data in tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader)):
            adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
            masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
            masked_nmr = batch_data["masked_nmr"].to(self.device)
            atom_labels = batch_data["atom_labels"].to(self.device)
            masked_flag = batch_data["masked_flag"].to(self.device)
            atom_pred = self.model(masked_mol_ids, masked_nmr, adjoin_matrix)
            loss = self.compute_loss_atom(atom_pred,atom_labels,masked_flag)
            self.optimizer.zero_grad()  #
            loss.backward()
            self.optimizer.step()
            cur_losses.append(loss.item())
            self.train_losses.append(loss.item())
            self.train_recovery.append(calculate_recovery(atom_pred.cpu(),atom_labels.cpu(),masked_flag.cpu()))
        self.train_losses += (cur_losses)

        metric = {
            "losses":np.array(cur_losses).mean(),
            "atom_recovery":np.array(self.train_recovery)[-1],
        }

        return metric


    def save_model(self,model_save_path):
        torch.save(self.model.state_dict(),model_save_path)

    def valid_test_iterations(self, mode='valid'):
        self.model.eval()
        if mode == 'test' :
            dataloader = self.test_dataloader
            losses = self.test_losses
            recovery = self.test_recovery
        elif mode =="valid":
            dataloader = self.valid_dataloader
            losses = self.valid_losses
            recovery = self.valid_recovery
        else:
            raise Exception("No such mode")
        cur_losses = []
        with torch.no_grad():

            for i, batch_data in enumerate(dataloader):
                adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
                masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
                masked_nmr = batch_data["masked_nmr"].to(self.device)
                atom_labels = batch_data["atom_labels"].to(self.device)
                masked_flag = batch_data["masked_flag"].to(self.device)
                atom_pred = self.model(masked_mol_ids, masked_nmr, adjoin_matrix)
                loss = self.compute_loss_atom(atom_pred, atom_labels, masked_flag)

                cur_losses.append(loss.cpu().data)
                recovery.append(calculate_recovery(atom_pred.cpu(),atom_labels.cpu(),masked_flag.cpu()))
        losses += cur_losses
        metric = {
            "losses":np.array(cur_losses).mean(),
            "atom_recovery":np.array(recovery)[-1],
        }
        return metric






