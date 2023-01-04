"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import time
import numpy as np
from sklearn import cluster
from tqdm import tqdm

from utils.kmeans import get_accuracy
from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader
from umap_embedding import reduce_dimensionality

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()
        
        
    def train_step_virtual(self, input_ids, attention_mask):
        
        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]
        
        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += 0.5*cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train_step_explicit(self, input_ids, attention_mask):
        
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Cluster loss
        feat1, feat2 = self.model.contrast_logits_cluster(embd2, embd3)
        contrast_cluster_loss = self.contrast_loss(feat1, feat2)
        losses["contrast_cluster_loss"] = contrast_cluster_loss['loss']
        loss = loss + self.eta * losses["contrast_cluster_loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        for i in tqdm(np.arange(self.args.max_iter+1)):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses = self.train_step_virtual(input_ids, attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                self.evaluate_embedding(i)
                self.model.train()

        return None   

    
    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))
        
        self.model.eval()
        for i, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
                    
        all_pred = all_prob.max(1)[1]
        acc_model = get_accuracy(all_labels.cpu(), all_pred.cpu())

        kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
        embeddings = all_embeddings.cpu().numpy()
        embeddings = reduce_dimensionality(np.asarray(embeddings))
        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
        # clustering accuracy 
        acc = get_accuracy(all_labels.cpu(), pred_labels.cpu())

        ressave = {"acc":acc, "acc_model":acc_model}
        # ressave.update(confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        def get_scores(target,pred):
            NMI = normalized_mutual_info_score(target,pred)
            ARI = adjusted_rand_score(target,pred)
            AMI = adjusted_mutual_info_score(target,pred)
            return {'NMI':NMI,'ARI':ARI,'AMI':AMI}


        # print('[Representation] Clustering scores:',get_scores(all_labels.cpu(), pred_labels.cpu())) 
        # print('[Representation] ACC: {:.3f}'.format(acc)) 
        print('[Model] Clustering scores:',get_scores(all_labels.cpu(), all_pred.cpu())) 
        print('[Model] ACC: {:.3f}'.format(acc_model))
        return None



            