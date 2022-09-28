from args import parse_args, set_seed
import numpy as np
from training.network import MultiNet
import time
import pandas as pd
import os
from flushed_print import print
from DTFT_MIL_train import MultiNet

if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(args.seed_value)


    acc = np.zeros((args.run, args.n_folds), dtype=float)
    precision = np.zeros((args.run, args.n_folds), dtype=float)
    recall = np.zeros((args.run, args.n_folds), dtype=float)
    auc = np.zeros((args.run, args.n_folds), dtype=float)
    loss = np.zeros((args.run, args.n_folds), dtype=float)

    csv_files = "csv_files"
    for irun in range(args.run):
        for csv_file in os.listdir(csv_files):

            fold_id = os.path.splitext(csv_file)[0].split("_")[1]

            references = pd.read_csv(os.path.join(csv_files, csv_file))

            train_bags = references["train"].apply(lambda x: x + ".h5").values.tolist()

            def func_val(x):
                value = None
                if isinstance(x, str):
                    value = x + ".h5"
                return value
            val_bags = references.apply(lambda row: func_val(row.val), axis=1).dropna().values.tolist()

            test_bags = references.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

            train_net = MultiNet(args)

            t1 = time.time()

            train_net.train(train_bags, val_bags, args, fold_id)

            test_net = MultiNet(args)

            acc[irun][int(fold_id)], auc[irun][int(fold_id)], precision[irun][int(fold_id)], recall[irun][int(fold_id)] = \
                test_net.predict(test_bags,args, fold_id,test_model=test_net.model)


    print('mean accuracy = ', np.mean(acc))
    print('std = ', np.std(acc))
    print(' mean precision = ', np.mean(precision))
    print('mean recall = ', np.mean(recall))
    print('std = ', np.std(recall))
    print(' mean auc = ', np.mean(auc))
    print('std = ', np.std(auc))






