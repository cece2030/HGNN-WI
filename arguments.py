import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--missing_link', type=float, default=0.5)
    parser.add_argument('--missing_feature', type=float, default=0.5)
    parser.add_argument('--train_per_class', type=int, default=20)
    parser.add_argument('--val_per_class', type=int, default=30)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)




    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_neighbor', type=int, default=5)
    parser.add_argument('--knn_metric', type=str, default='cosine', choices=['cosine','minkowski'])
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')




    parser.add_argument("--classifier_epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--classifier_lr", type=float, default=0.01,
                        help="classifier learning rate")
    parser.add_argument("--classifier_weight_decay", type=float, default=0.05,
                        help="classifier learning rate")
    parser.add_argument('--SemanticAttention_hidden', type=int, default=64)


    return parser.parse_args()