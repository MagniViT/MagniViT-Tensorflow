from args import parse_args, set_seed
import numpy as np
import pandas as pd
import os
from flushed_print import print
from training.DTFT_train import MultiNet

if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(args.seed_value)


    acc = np.zeros((args.run, args.n_folds), dtype=float)
    f_score = np.zeros((args.run, args.n_folds), dtype=float)
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

            train_net = MultiNet(args, training_flag=True)

            train_net.train(train_bags, val_bags, fold_id)

            test_net = MultiNet(args, training_flag=False)

            acc[irun][int(fold_id)], auc[irun][int(fold_id)], precision[irun][int(fold_id)], recall[irun][int(fold_id)] ,f_score[irun][int(fold_id)]= \
                test_net.predict(test_bags, fold_id,tier_1=test_net.model_1,tier_2=test_net.model_2 )


    print('mean accuracy = ', np.mean(acc))
    print('std = ', np.std(acc))
    print(' mean precision = ', np.mean(precision))
    print('std = ', np.std(precision))
    print('mean recall = ', np.mean(recall))
    print('std = ', np.std(recall))
    print(' mean auc = ', np.mean(auc))
    print('std = ', np.std(auc))
    print(' mean f_score = ', np.mean(f_score))
    print('std = ', np.std(f_score))






