"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import csv
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
from metrics import accuracy_TU as accuracy, precision, recall, f1, roc_auc, accuracy_all_classes
from captum.attr import IntegratedGradients
from functools import partial

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels,batch_llms) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_llms = batch_llms.to(device)
        # batch_labels = batch_labels.to(device)
        batch_labels = batch_labels.to(device).long()
        optimizer.zero_grad()
        if model.name in ["PRGNN", "LINet"]:
            batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
            batch_labels = batch_labels.long()
            loss = model.loss(batch_scores, batch_labels, score1, score2)
        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e,batch_llms)
            loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels,batch_llms) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_llms = batch_llms.to(device)

            # batch_labels = batch_labels.to(device)
            batch_labels = batch_labels.to(device).long()

            if model.name in ["PRGNN", "LINet"]:
                batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
                loss = model.loss(batch_scores, batch_labels, score1, score2)
            else:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
                loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc


def evaluate_network_all_metric(model, device, data_loader, epoch=0, path=''):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    labels = []

    def saliency_forward(x, e, llm, g):
        return model.forward(g, x, e, llm)

    with torch.no_grad():
        y_pred = []
        y_pred_probs = []
        maps = []
        for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):
            # batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device).long()
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_llms = batch_llms.to(device)

            batch_labels = batch_labels.to(device)
            if model.name in ["PRGNN", "LINet"]:
                batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
                loss = model.loss(batch_scores, batch_labels, score1, score2)
            else:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
                loss = model.loss(batch_scores, batch_labels)

                # generate saliency maps
                if path:
                    ig = IntegratedGradients(partial(saliency_forward, e=batch_e, llm=batch_llms, g=batch_graphs))
                    saliency = ig.attribute(batch_x, n_steps=1, target=0)
                    maps.append(saliency.reshape(-1, 116, 116))

            labels.append(batch_labels.detach().cpu())
            y_pred.append(batch_scores.detach().cpu())
            y_pred_probs.append(torch.softmax(batch_scores, dim=1).detach().cpu())

            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

        y_true = torch.cat(labels, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_pred_probs = torch.cat(y_pred_probs, dim=0).numpy()

        if len(np.unique(y_true)) == 2:
            test_precision = precision(y_pred, y_true)
            test_recall = recall(y_pred, y_true)
            test_f1 = f1(y_pred, y_true)
            test_roc_auc = roc_auc(y_pred, y_true)
        else:
            test_precision = epoch_test_acc
            test_recall = epoch_test_acc
            test_f1 = epoch_test_acc
            test_roc_auc = epoch_test_acc

        all_acc = accuracy_all_classes(y_pred, y_true)

        if path:
            maps = torch.cat(maps, dim=0)
            torch.save(maps, path)

    return epoch_test_loss, epoch_test_acc, test_precision, test_recall, test_f1, test_roc_auc, all_acc


"""
    DEBUGGING FUNCTION: Test where the leakage is coming from
"""
def diagnose_data_leakage(model, device, data_loader, model_name):
    """
    Test 1: Model accuracy with only graph features (no LLM)
    Test 2: Model accuracy with only LLM embeddings (no graph)
    Test 3: Combined accuracy
    """
    model.eval()
    
    print("\n" + "="*70)
    print("LEAKAGE DIAGNOSIS: Testing model component contribution")
    print("="*70)
    
    with torch.no_grad():
        test_acc_combined = 0
        test_acc_graph_only = 0
        test_acc_llm_only = 0
        nb_data = 0
        
        for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_llms = batch_llms.to(device)
            batch_labels = batch_labels.to(device).long()
            
            if model_name not in ["PRGNN", "LINet"]:
                # Test 1: Combined (normal)
                batch_scores_combined = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
                test_acc_combined += accuracy(batch_scores_combined, batch_labels)
                
                # Test 2: Only graph (pass zero embeddings)
                batch_llms_zero = torch.zeros_like(batch_llms)
                try:
                    batch_scores_graph_only = model.forward(batch_graphs, batch_x, batch_e, batch_llms_zero)
                    test_acc_graph_only += accuracy(batch_scores_graph_only, batch_labels)
                except Exception as e:
                    print(f"Cannot test graph-only: {e}")
                
                # Test 3: Only LLM (create dummy graph - all features zero)
                batch_x_zero = torch.zeros_like(batch_x)
                try:
                    batch_scores_llm_only = model.forward(batch_graphs, batch_x_zero, batch_e, batch_llms)
                    test_acc_llm_only += accuracy(batch_scores_llm_only, batch_labels)
                except Exception as e:
                    print(f"Cannot test LLM-only: {e}")
                
                nb_data += batch_labels.size(0)
        
        test_acc_combined /= (iter + 1)
        test_acc_graph_only /= (iter + 1)
        test_acc_llm_only /= (iter + 1)
        
        print(f"\nResults on TEST set ({nb_data} samples):")
        print(f"  Combined (Graph + LLM):     {test_acc_combined:.4f}")
        print(f"  Graph only (LLM=0):         {test_acc_graph_only:.4f}")
        print(f"  LLM only (Graph feats=0):   {test_acc_llm_only:.4f}")
        
        print("\nINTERPRETATION:")
        if test_acc_llm_only > 0.9 and test_acc_graph_only < 0.6:
            print("  🚨 CRITICAL LEAKAGE: Model relies almost entirely on LLM embeddings!")
            print("     → LLM embeddings encode label information directly")
            print("     → CHECK: How are label_prompts generated?")
        elif test_acc_graph_only > 0.9 and test_acc_llm_only < 0.6:
            print("  ✓ Model primarily uses graph structure (less leakage)")
        elif test_acc_combined > 0.95 and test_acc_graph_only < 0.7 and test_acc_llm_only < 0.7:
            print("  ⚠️  Both components weak individually but perfect together")
            print("     → Possible overfitting to training distribution")
        elif test_acc_combined < 0.75:
            print("  ✓ Reasonable accuracy - likely legitimate learning")
        
        print("="*70 + "\n")
        
        return test_acc_combined, test_acc_graph_only, test_acc_llm_only


"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        
        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels) 
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_dense(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc


def check_patience(all_losses, best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter
