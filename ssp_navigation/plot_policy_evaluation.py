import numpy as np
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

parser = argparse.ArgumentParser(
    'Combine csv files if neccessary and plot results'
)

parser.add_argument('--folder', type=str, default='eval_data')
parser.add_argument('--fname', type=str, default='combined_data.csv')
parser.add_argument('--best-epoch', action='store_true')

args = parser.parse_args()

combined_fname = os.path.join(args.folder, args.fname)

if os.path.isfile(combined_fname) and False:
# if os.path.isfile(combined_fname):
    # combined file already exists, load it
    df = pd.read_csv(combined_fname)
else:
    # combined file does not exist, create it
    files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]

    df = pd.DataFrame()

    for file in files:
        if file == args.fname:
            continue
        df_temp = pd.read_csv(os.path.join(args.folder, file))

        # if 'SSP Scaling' not in df_temp:
        #     df_temp['SSP Scaling'] = 0.5

        # only want to look at the correct sigma results in this case
        if 'bounds_exps' in args.folder:
            # if np.any(df_temp['Encoding'] == 'pc-gauss') and np.any(df_temp['Sigma'] != 0.375):
            if ('pc-gauss' in file) and ('scaling' in file):
                continue
            else:
                df = df.append(df_temp)
        else:

            df = df.append(df_temp)

    df.to_csv(combined_fname)


if args.best_epoch:
    # Consider only the number of epochs trained for that result in the best test RMSE
    columns = [
        'Dimensionality',
        'Hidden Layer Size',
        'Hidden Layers',
        'Encoding',
        'Seed',
        'Maze ID Type',

        # # Other differentiators
        # 'Number of Mazes Tested',
        # 'Number of Goals Tested',

        # Command line supplied tags
        'Dataset',
        # 'Trained On',
        # 'Epochs',  # this will be used to compute the max over
        'Batch Size',
        'Number of Mazes',
        'Loss Function',
        'Learning Rate',
        'Momentum',
        # 'Sigma',
        # 'Hex Freq Coef',
    ]

    unique_column_dict = {}

    for column in columns:
        unique_column_dict[column] = df[column].unique()

    ordered_column_names = list(unique_column_dict.keys())

    # New dataframe containing only the best early-stop rows
    df_new = pd.DataFrame()

    for column_spec in product(*[unique_column_dict[column] for column in ordered_column_names]):
        # print(ordered_column_names)
        # print(column_spec)
        df_tmp = df
        # load only the columns that match the current spec
        for i in range(len(ordered_column_names)):
            # print("{} == {}".format(ordered_column_names[i], column_spec[i]))
            df_tmp = df_tmp[df_tmp[ordered_column_names[i]] == column_spec[i]]

        if df_tmp.empty:
            # It is possible there is no data for this combination. If so, skip it
            continue
        else:
            # Only difference between rows is the epochs trained for.
            # Find the one with the best result (i.e. not overfit or underfit, best time to early-stop)
            df_new = df_new.append(df_tmp.loc[df_tmp['Angular RMSE'].idxmin()])

    # overwrite old dataframe with the new one
    df = df_new

    print(df)


# df = df[df['Number of Mazes'] == 10]
# df = df[df['Dimensionality'] == 256]
# df = df[df['Number of Mazes'] == 20]
# df = df[df['Dimensionality'] == 512]
# df = df[df['Number of Mazes'] == 5]
# df = df[df['Number of Mazes'] == 64]

if False:
    # TODO: sort these out
    # df = df[df['Dimensionality'] == 128]
    # df = df[df['Encoding'] == 'ssp']
    #
    # print(df)

    # df = df[df['Encoding'] == 'frozen-learned']
    #
    # plt.figure()
    # sns.barplot(data=df, x='Dataset', y='RMSE', hue='Trained On')



    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='RMSE')
    # plt.figure()
    # sns.barplot(data=df, x='Dimensionality', y='RMSE')
    #
    # plt.figure()
    # sns.barplot(data=df, x='Dataset', y='RMSE', hue='Encoding')



    plt.figure()
    # sns.barplot(data=df, x='Dataset', y='RMSE', hue='Number of Mazes')
    # sns.barplot(data=df, x='Dataset', y='Angular RMSE', hue='Number of Mazes')


    # sns.barplot(data=df, x='Dimensionality', y='Angular RMSE', hue='Batch Size')

    # sns.barplot(data=df, x='Dimensionality', y='Angular RMSE', hue='Loss Function')


elif 'debugging_eval_data' in args.folder:

    plt.figure()
    sns.barplot(data=df, x='Dimensionality', y='Angular RMSE')

    plt.figure()
    sns.barplot(data=df, x='Number of Mazes', y='Angular RMSE')

    plt.figure()
    sns.barplot(data=df, x='Dimensionality', y='Train Angular RMSE')

    plt.figure()
    sns.barplot(data=df, x='Number of Mazes', y='Train Angular RMSE')

elif 'lr_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Dimensionality', y='Angular RMSE', hue='Learning Rate')
    plt.figure()
    sns.barplot(data=df, x='Dimensionality', y='Angular RMSE', hue='Momentum')


    plt.figure()
    sns.barplot(data=df, x='Dimensionality', y='Train Angular RMSE', hue='Learning Rate')
    plt.figure()
    sns.barplot(data=df, x='Dimensionality', y='Train Angular RMSE', hue='Momentum')


elif 'loss_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Dimensionality', y='Angular RMSE', hue='Loss Function')

    # plt.figure()
    # sns.barplot(data=df, x='Dimensionality', y='Train Angular RMSE', hue='Loss Function')

elif 'large_map_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')


    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Dimensionality')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Number of Mazes')

elif 'network_size_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Hidden Layers', y='Angular RMSE', hue='Hidden Layer Size')

    plt.figure()
    sns.barplot(data=df, x='Hidden Layers', y='Train Angular RMSE', hue='Hidden Layer Size')

elif 'cap_enc_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')

    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE')

    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')


    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Number of Mazes')

    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE')

    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Dimensionality')


    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='RMSE', hue='Number of Mazes')
    #
    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='RMSE')
    #
    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='RMSE', hue='Dimensionality')
    #
    #
    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='Train RMSE', hue='Number of Mazes')
    #
    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='Train RMSE')
    #
    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='Train RMSE', hue='Dimensionality')

elif 'pcgauss_sigma_exps' in args.folder:

    plt.figure()
    # sns.barplot(data=df, x='Sigma', y='Angular RMSE')
    sns.barplot(data=df, x='Dimensionality', y='Angular RMSE')
elif 'ssp_scaling_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='SSP Scaling')


    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='SSP Scaling')
elif 'grid_ssp_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='SSP Scaling')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')


    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='SSP Scaling')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Number of Mazes')
else:
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')

    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='SSP Scaling')


    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Number of Mazes')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Dimensionality')

try:
    print("Training Angular RMSE:", df['Train Angular RMSE'].mean())
    print("Testing Angular RMSE:", df['Angular RMSE'].mean())
except KeyError:
    print("train/test error distinction not included in dataset")

plt.show()
