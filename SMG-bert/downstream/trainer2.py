import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from easydict import EasyDict
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import get_downstream_dataloader
from utils import get_logger, get_angle, get_torsion
class Downstream_Trainer():
    def __init__(self, config, model):
        self.config = config
        self.device = config.device
        self.model = model.to(self.config.device)
        base_name = os.path.basename(self.config.res_dir)
        res_dir = os.path.join(self.config.res_dir,config.dataset.split_type,str(config.seed))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        self.classify_loss = nn.BCEWithLogitsLoss()
        self.config.log_path = os.path.join(res_dir,base_name+".log")
        self.config.res_path = os.path.join(res_dir,base_name+".csv")
        self.config.best_model_path = os.path.join(res_dir, base_name + ".pth")

        self.logger = get_logger(config.log_path)
        self.logger.info(f"model_save_dir:{self.config.res_dir},log_path:{self.config.log_path},res_path:{self.config.res_path}, config{config}")
        self.focal_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.01,0.99]).to(self.device))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.AdamW.lr,
                                           eps=self.config.train.AdamW.eps,
                                           betas=(self.config.train.AdamW.beta1, self.config.train.AdamW.beta2))


        self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_downstream_dataloader(config,self.logger,smiles_data_path=config.dataset.smiles_data_path,dist_data_path=config.dataset.dist_data_path,
                                                                                                     angle_data_path=config.dataset.angle_data_path,
                                                                                                     torsion_data_path=config.dataset.torsion_data_path,label_data_path=config.dataset.label_data_path)




        self.best_downstream_metric = 0
        self.best_epoch = 0

        self.valid_downstream_metric_list = []


        self.theta_atom_1D = torch.tensor(1., requires_grad=True)
        self.theta_atom_2D = torch.tensor(1., requires_grad=True)
        self.theta_atom_3D = torch.tensor(1., requires_grad=True)
        self.theta_adj = torch.tensor(1., requires_grad=True)
        self.theta_dist = torch.tensor(1., requires_grad=True)
        self.theta_angle = torch.tensor(1., requires_grad=True)
        self.theta_torsion = torch.tensor(1., requires_grad=True)


        self.train_step = 0
        self.val_step = 0

    def train(self, epochs=50):
        train_metric_list = []
        valid_metric_list = []
        early_stop_num = 0
        for epoch in range(epochs):
            train_metric = self.train_iterations()
            valid_metric = self.valid_test_iterations()
            self.logger.info(f"epoch:{epoch},train_metric:{train_metric},valid_metric:{valid_metric}")

            train_metric_list.append(train_metric)
            valid_metric_list.append(valid_metric)

            if valid_metric.loss == np.min(np.array([valid_metric.loss for valid_metric in valid_metric_list])):
                early_stop_num = 0
                test_metric = self.valid_test_iterations(mode="test")
                self.best_epoch = epoch
                self.best_downstream_metric = test_metric
                self.logger.info(f"best_epoch:{self.best_epoch},best_downstream_metric:{self.best_downstream_metric}")
                self.save_model(self.config.best_model_path)
            else:
                early_stop_num += 1


            if early_stop_num > 5:
                break


        self.logger.info(f"final best_epoch:{self.best_epoch},best_downstream_metric:{self.best_downstream_metric}")
        if self.config.bert.task_type == "classify":
            res = pd.DataFrame({"best_epoch": [self.best_epoch], "roc_auc": [self.best_downstream_metric.roc_auc]})
        else:
            res = pd.DataFrame({"best_epoch": [self.best_epoch], "r2": [self.best_downstream_metric.r2],"rmse": [self.best_downstream_metric.rmse]})
        res.to_csv(self.config.res_path)

    def train_iterations(self):
        self.model.train()
        losses_total = []
        label_list = []
        output_list = []
        for tra_step, batch_data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):

            self.train_step += 1
            atom_id_raw = batch_data['atom_id_raw'].to(self.device)
            atom_id_mask = batch_data['atom_id_mask'].to(self.device)



            adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
            dist_matrix = batch_data["dist_matrix"].to(self.device)
            dist_i = batch_data["dist_i"].to(self.device)
            dist_j = batch_data["dist_j"].to(self.device)
            masked_flag = batch_data["masked_flag"].to(self.device)
            angle = batch_data["angle"].to(self.device)
            angle_i = batch_data["angle_i"].to(self.device)
            angle_j = batch_data["angle_j"].to(self.device)
            angle_k = batch_data["angle_k"].to(self.device)
            torsion_k = batch_data["torsion_k"].to(self.device)
            torsion_i = batch_data["torsion_i"].to(self.device)
            torsion_j = batch_data["torsion_j"].to(self.device)
            torsion_t = batch_data["torsion_t"].to(self.device)
            torsion = batch_data["torsion"].to(self.device)
            dist_mask = batch_data["dist_mask"].to(self.device)
            label = batch_data["label"].to(self.device)
            if self.config.train.use_gnn_adj:
                gnn_adj = batch_data["gnn_adj"].to(self.device)
            else:
                gnn_adj = None

            adj_pred, pos_pred, x_1D,x_2D,x_3D, share,label_pred = self.model(atom_id_raw,dist_i,dist_j,mask=adjoin_matrix,gnn_adj=gnn_adj,dist_score=dist_matrix)

            label_pred = label_pred.squeeze(-1)

            if self.config.bert.task_type == "classify":
                loss_downstream = self.classify_loss(label_pred,label.float())
            else:
                loss_downstream = torch.nn.functional.mse_loss(label_pred,label)


            loss_atom_1D=loss_atom_2D=loss_atom_3D=loss_adj=loss_dist=loss_angle=loss_torsion = torch.tensor(0)



            if self.config.loss.adj_loss:
                adj_label = adjoin_matrix.masked_select(dist_mask).unsqueeze(1)
                loss_adj = self.focal_loss(adj_pred, adj_label)


            if self.config.loss.dist_loss:
                loss_dist = Downstream_Trainer.compute_loss_dist(dist_matrix, pos_pred, dist_i, dist_j,dist_mask)
            if self.config.loss.angle_loss:
                loss_angle = Downstream_Trainer.compute_loss_angle(angle, pos_pred, angle_i, angle_j, angle_k)
            if self.config.loss.torsion_loss:
                loss_torsion = Downstream_Trainer.compute_loss_torsion(torsion, pos_pred, torsion_k, torsion_i, torsion_j, torsion_t)

            if self.config.loss.agg_method == "add":
                loss = loss_atom_1D + loss_atom_2D+loss_atom_3D+loss_adj+loss_dist+loss_angle+loss_torsion
            else:
                loss = 1 / (self.theta_atom_1D ** 2) * loss_atom_1D \
                       + 1 / (self.theta_atom_2D ** 2) * loss_atom_2D \
                       + 1 / (self.theta_atom_3D ** 2) * loss_atom_3D\
                       + 1 /(self.theta_adj ** 2) * loss_adj \
                       + 1 / ( 2 * self.theta_dist ** 2) * loss_dist \
                       + 1 / (2 * self.theta_angle ** 2) * loss_angle + \
                       1 / (2 * self.theta_torsion ** 2) * loss_torsion + \
                       torch.log(self.theta_atom_1D * self.theta_atom_2D*self.theta_atom_3D *self.theta_adj* self.theta_dist * self.theta_angle * self.theta_torsion)

            loss_total = loss_downstream + self.config.loss.alpha * loss

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            label_list += label.cpu().numpy().tolist()
            output_list += label_pred.cpu().detach().numpy().tolist()
            losses_total.append(loss_total.item())

        if self.config.bert.task_type == "classify":
            downstream_metric = {"loss":np.array(losses_total).mean(),  "roc_auc": roc_auc_score(label_list, output_list)}
        else:
            downstream_metric = {"loss":np.array(losses_total).mean(), "rmse": mean_squared_error(label_list, output_list,squared=False),"r2":r2_score(label_list, output_list)}
        return EasyDict(downstream_metric)

    def valid_test_iterations(self, mode='valid'):
        self.model.eval()

        losses_total = []
        output_list = []
        label_list = []

        if mode == 'test':
            dataloader = self.test_dataloader
        elif mode == "valid":
            dataloader = self.valid_dataloader
        else:
            raise Exception("No such mode")

        with torch.no_grad():

            for i, batch_data in tqdm(enumerate(dataloader),total=len(dataloader)):
                self.val_step += 1
                atom_id_raw = batch_data['atom_id_raw'].to(self.device)
                atom_id_mask = batch_data['atom_id_mask'].to(self.device)
                adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
                dist_matrix = batch_data["dist_matrix"].to(self.device)
                dist_i = batch_data["dist_i"].to(self.device)
                dist_j = batch_data["dist_j"].to(self.device)
                masked_flag = batch_data["masked_flag"].to(self.device)
                angle = batch_data["angle"].to(self.device)
                angle_i = batch_data["angle_i"].to(self.device)
                angle_j = batch_data["angle_j"].to(self.device)
                angle_k = batch_data["angle_k"].to(self.device)
                torsion_k = batch_data["torsion_k"].to(self.device)
                torsion_i = batch_data["torsion_i"].to(self.device)
                torsion_j = batch_data["torsion_j"].to(self.device)
                torsion_t = batch_data["torsion_t"].to(self.device)
                torsion = batch_data["torsion"].to(self.device)
                dist_mask = batch_data["dist_mask"].to(self.device)
                label = batch_data["label"].to(self.device)
                if self.config.train.use_gnn_adj:
                    gnn_adj = batch_data["gnn_adj"].to(self.device)
                else:
                    gnn_adj = None

                adj_pred, pos_pred, x_1D, x_2D, x_3D, share, label_pred = self.model(
                    atom_id_raw, dist_i, dist_j, mask=adjoin_matrix, gnn_adj=gnn_adj, dist_score=dist_matrix)

                label_pred = label_pred.squeeze(-1)

                if self.config.bert.task_type == "classify":
                    loss_downstream = self.classify_loss(label_pred, label.float())
                else:
                    loss_downstream = torch.nn.functional.mse_loss(label_pred, label)

                loss_atom_1D = loss_atom_2D = loss_atom_3D = loss_adj = loss_dist = loss_angle = loss_torsion = torch.tensor(
                    0)

                if self.config.loss.adj_loss:
                    adj_label = adjoin_matrix.masked_select(dist_mask).unsqueeze(1)
                    loss_adj = self.focal_loss(adj_pred, adj_label)


                if self.config.loss.dist_loss:
                    loss_dist = Downstream_Trainer.compute_loss_dist(dist_matrix, pos_pred, dist_i, dist_j,dist_mask)
                if self.config.loss.angle_loss:
                    loss_angle = Downstream_Trainer.compute_loss_angle(angle, pos_pred, angle_i, angle_j, angle_k)
                if self.config.loss.torsion_loss:
                    loss_torsion = Downstream_Trainer.compute_loss_torsion(torsion, pos_pred, torsion_k, torsion_i, torsion_j, torsion_t)

                if self.config.loss.agg_method == "add":
                    loss = loss_atom_1D + loss_atom_2D+loss_atom_3D+loss_adj+loss_dist+loss_angle+loss_torsion
                else:
                    loss = 1 / (self.theta_atom_1D ** 2) * loss_atom_1D \
                           + 1 / (self.theta_atom_2D ** 2) * loss_atom_2D \
                           + 1 / (self.theta_atom_3D ** 2) * loss_atom_3D\
                           + 1 /(self.theta_adj ** 2) * loss_adj \
                           + 1 / ( 2 * self.theta_dist ** 2) * loss_dist \
                           + 1 / (2 * self.theta_angle ** 2) * loss_angle + \
                           1 / (2 * self.theta_torsion ** 2) * loss_torsion + \
                           torch.log(self.theta_atom_1D * self.theta_atom_2D*self.theta_atom_3D *self.theta_adj* self.theta_dist * self.theta_angle * self.theta_torsion)

                loss_total = loss_downstream + self.config.loss.alpha * loss

                label_list += label.cpu().numpy().tolist()
                output_list += label_pred.cpu().detach().numpy().tolist()
                losses_total.append(loss_total.item())

            if self.config.bert.task_type == "classify":
                downstream_metric = {"loss":np.array(losses_total).mean(),  "roc_auc": roc_auc_score(label_list, output_list)}
            else:
                downstream_metric = {"loss":np.array(losses_total).mean(), "rmse": mean_squared_error(label_list, output_list,squared=False),"r2":r2_score(label_list, output_list)}
            return EasyDict(downstream_metric)


    @staticmethod
    def compute_loss_dist(dist_labels, pos, dist_i, dist_j,dist_mask):
        dist_labels = dist_labels[:,:dist_mask.shape[1],:dist_mask.shape[1]].masked_select(dist_mask)
        pos = pos.reshape(-1,pos.shape[-1])
        dist_pred = ((pos[dist_i] - pos[dist_j]).pow(2).sum(dim=-1)+1e-9).sqrt()
        loss = torch.nn.functional.mse_loss(dist_pred, dist_labels)
        return loss

    @staticmethod
    def compute_loss_torsion(torsion_labels, pos, torsion_k, torsion_i, torsion_j, torsion_t):
        pos = pos.reshape(-1, pos.shape[-1])
        torsion_pred = get_torsion(pos, torsion_k, torsion_i, torsion_j, torsion_t)
        loss = torch.nn.functional.mse_loss(torsion_pred, torsion_labels)
        return loss

    @staticmethod
    def compute_loss_angle(angle_labels, pos, angle_i, angle_j, angle_k):
        pos = pos.reshape(-1, pos.shape[-1])
        angle_pred = get_angle(pos, angle_i, angle_j, angle_k)
        loss = torch.nn.functional.mse_loss(angle_pred, angle_labels)
        return loss

    @staticmethod
    def compute_loss_atom(atom_pred, atom_labels, masked_flag):
        atom_pred = atom_pred[masked_flag == 1]
        atom_labels = atom_labels[masked_flag == 1]
        loss_atom = torch.nn.functional.cross_entropy(atom_pred, atom_labels)
        return loss_atom

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)
