import datetime

import torch

import arguments
import time

from utils import *
from models import model_attention, LogReg


from sklearn.metrics import f1_score
args = arguments.parse_args()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def main():

    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
        val_mask, test_mask, meta_paths = load_dblp(args.missing_feature)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
    labels = labels.to(device)
    idx_train = train_mask.to(device)
    idx_val = val_mask.to(device)
    idx_test = test_mask.to(device)
    x = features[0]
    semantic_embeddings = process_adj_features(g, meta_paths, features, args)
    # features_list = [mat2tensor(features).to(device)
    #                  for features in features_list]

    semantic_embeddings = [features.to(device)
                     for features in semantic_embeddings]
    semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
    meta_adj_list = create_meta_adj_list(g, features, meta_paths)


    meta_adj_list = [adj.to(device)
                           for adj in meta_adj_list]
    adj_u = union_adj_matrices(meta_adj_list)
    model = model_attention(features[0].shape[1], args.hidden, activation='prelu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def train_model():

        acc_results = []
        sparse = True

        memory0 = show_info()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        # 训练的过程
        for epoch in range(args.epochs):
            t_start = time.time()

            model.train()
            optimizer.zero_grad()

            lbl_1 = torch.ones(features[0].shape[0])
            lbl_2 = torch.zeros(features[0].shape[0])
            lbl_3 = torch.ones(features[0].shape[0])
            lbl = torch.cat((lbl_1, lbl_2, lbl_3), 0)
            lbl = lbl.to(device)
            x, x_aug, adj_knn, adj,logits_1 = model(semantic_embeddings, adj_u, sparse=True)

            loss_train = F.binary_cross_entropy_with_logits(logits_1, lbl)

            # tag = str(int(np.random.random() * 10000000000))

            best_model_state_dict = None
            if loss_train < best:
                best = loss_train
                best_t = epoch
                cnt_wait = 0
                best_model_state_dict = model.state_dict()
                # torch.save(model.state_dict(), 'pkl/best_model' + tag + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break
            # model.zero_grad()
            loss_train.backward()
            optimizer.step()
            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, loss_train.item(), t_end - t_start))
        memory1 = show_info()
        print('Memory {:.4f}'.format(memory1 - memory0))
        print('The best epoch is: ', best_t)
        or_embeds,pr_embeds = model.embed(x_aug, adj_knn, sparse)
        embed0,embed1 = model.embed(x,adj,sparse)
        embeds = pr_embeds + embed1 +or_embeds + embed0

        train_embs = embeds[0, idx_train == True]
        val_embs = embeds[0, idx_val == True]
        test_embs = embeds[0, idx_test == True]
        # torch.save(train_embs, 'train_embs.pth')
        # torch.save(val_embs, 'val_embs.pth')
        # torch.save(test_embs, 'test_embs.pth')
        # train_lbls = torch.argmax(client_data[j].y[data.train_mask == True], dim=1)
        # train_embs = torch.load('train_embs.pth')
        # val_embs = torch.load('val_embs.pth')
        # test_embs = torch.load('test_embs.pth')


        train_lbls = labels[idx_train].to(device)

        val_lbls = labels[idx_val].to(device)
        test_lbls = labels[idx_test].to(device)

        for _ in range(50):  # 微调的过程
            log = LogReg(train_embs.shape[1], num_classes)
            log = log.to(device)
            opt = torch.optim.Adam(log.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
            val_accs = []
            val_micro_f1s = []
            test_micro_f1s = []
            test_micro_f1s_result = []
            val_macro_f1s = []
            test_macro_f1s = []
            test_macro_f1s_result = []
            for _ in range(args.classifier_epochs):
                # train
                log.train()
                opt.zero_grad()
                logits = log(train_embs)

                loss = F.cross_entropy(logits, train_lbls)

                loss.backward()
                opt.step()

            log.eval()
            with torch.no_grad():
                # val
                logits = log(val_embs)
                preds = torch.argmax(logits, dim=1)
                val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
                val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
                val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')
                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)
            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
            test_macro_f1s.append(torch.tensor(test_f1_macro * 100))
            test_micro_f1s.append(torch.tensor(test_f1_micro * 100))
        test_macro_f1s = torch.stack(tuple(test_macro_f1s))
        test_micro_f1s = torch.stack(tuple(test_micro_f1s))
        test_macro_f1s_result.append(test_macro_f1s.mean().numpy())
        test_micro_f1s_result.append(test_micro_f1s.mean().numpy())

        # Print the final results
        print("Final Test F1 Macro: {:.4f}".format(test_macro_f1s_result[0]))
        print("Final Test F1 Micro: {:.4f}".format(test_micro_f1s_result[0]))
        return (test_macro_f1s_result), (test_micro_f1s_result)

    return train_model()

if __name__ == "__main__":

    accs = []
    ma_f1s = []
    mi_f1s = []
    for trial in range(args.num_trials):
        setup_seed(trial)
        ma_f1, mi_f1 = main()
        ma_f1s.append(ma_f1)
        mi_f1s.append(mi_f1)
    avg_ma = np.mean(ma_f1s)
    std_ma = np.std(ma_f1s)
    avg_mi = np.mean(mi_f1s)
    std_mi = np.std(mi_f1s)
    print('[FINAL RESULT] AVG_ma_f1:{:.2f}+-{:.2f}'.format(avg_ma, std_ma))
    print('[FINAL RESULT] AVG_mi_f1:{:.2f}+-{:.2f}'.format(avg_mi, std_mi))