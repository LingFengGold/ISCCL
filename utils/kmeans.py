"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def get_mean_embeddings(bert, input_ids, attention_mask):
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat

def get_accuracy(labels, prediction):
    
    cm = confusion_matrix(labels, prediction)

    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    def linear_assignment(cost_matrix):    
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy 

def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(tqdm(train_loader)):

        text, label = batch['text'], batch['label']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)
        
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings.detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().numpy()), axis=0)

    
    # Perform KMeans clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))

    # confusion.add(pred_labels, true_labels)
    # confusion.optimal_assignment(num_classes)

    true_labels -= 1 #To be in the same range 0-3 instead of 1-4
    accuracy = get_accuracy(true_labels, pred_labels)

    # print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, accuracy, clustering_model.cluster_centers_.shape))
    
    return clustering_model.cluster_centers_



