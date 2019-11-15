import numpy as np
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    'Combine csv files if neccessary and plot results'
)

parser.add_argument('--folder', type=str, default='eval_data')
parser.add_argument('--fname', type=str, default='combined_data.csv')

args = parser.parse_args()

combined_fname = os.path.join(args.folder, args.fname)

if os.path.isfile(combined_fname) and False:
    # combined file already exists, load it
    df = pd.read_csv(combined_fname)
else:
    # combined file does not exist, create it
    files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]

    df = pd.DataFrame()

    for file in files:
        df_temp = pd.read_csv(os.path.join(args.folder, file))

        df = df.append(df_temp)

    df.to_csv(combined_fname)


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


elif 'large_map_exps' in args.folder:
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE')
    plt.figure()
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')


    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Dimensionality')
    # plt.figure()
    # sns.barplot(data=df, x='Encoding', y='Train Angular RMSE')

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

print("Training Angular RMSE:", df['Train Angular RMSE'].mean())
print("Testing Angular RMSE:", df['Angular RMSE'].mean())

plt.show()
