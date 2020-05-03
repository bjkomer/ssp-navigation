import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser('Generate final version of figures for thesis')

parser.add_argument('--figure', type=str, default='capacity',
    choices=[
        'capacity', 'bounds', 'connected-tiled-maze', '100-tiled-maze', '25-tiled-maze', 'tiled-maze',
        'blocksmaze', 'small-env', 'large-env', 'network-size', 'hidden-size', 'dim'
    ]
)

args = parser.parse_args()

old_names = [
    'hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot',
    'learned', 'learned-normalized' '2d', 'random',
    'frozen-learned', 'frozen-learned-normalized',
]
order = [
    'Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot',
    'Learned', 'Learned Normalized', '2D', 'Random',
    'Frozen Learned', 'Frozen Learned Normalized',
]

if args.figure == 'capacity':
    folders = ['eval_data_tt/final_capacity_exps/']
elif args.figure == 'connected-tiled-maze':
    folders = ['eval_data_tt/med_dim_connected_tiledmaze_more_samples_exps']
elif args.figure == '100-tiled-maze':
    folders = ['eval_data_tt/large_dim_100tiledmaze_more_samples_exps']
elif args.figure == '25-tiled-maze':
    folders = ['eval_data_tt/large_dim_25tiledmaze_more_samples_exps']
elif args.figure == 'tiled-maze':
    folders = ['eval_data_tt/large_dim_25tiledmaze_more_samples_exps', 'eval_data_tt/large_dim_100tiledmaze_more_samples_exps']
elif args.figure == 'blocksmaze':
    folders = ['eval_data_tt/final_blocksmaze_exps']
    df_fresh = pd.DataFrame()
elif args.figure == 'small-env':
    folders = ['eval_data_tt/small_map_med_dim_exps']
elif args.figure == 'large-env':
    folders = ['eval_data_tt/large_map_med_dim_exps']
elif args.figure == 'network-size':
    folders = ['eval_data_tt/network_size_exps']
elif args.figure == 'hidden-size':
    folders = ['eval_data_tt/final_hidsize_exps']
elif args.figure == 'dim':
    folders = ['eval_data_tt/final_dim_exps']
else:
    raise NotImplementedError

df = pd.DataFrame()

for folder in folders:

    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    for file in files:
        df_temp = pd.read_csv(os.path.join(folder, file))

        # if 'SSP Scaling' not in df_temp:
        #     df_temp['SSP Scaling'] = 0.5

        # only want to look at the correct sigma results in this case
        if 'bounds_exps' in folder:
            # if np.any(df_temp['Encoding'] == 'pc-gauss') and np.any(df_temp['Sigma'] != 0.375):
            if ('pc-gauss' in file) and ('scaling' in file):
                continue
            else:
                df = df.append(df_temp)
        # elif 'dataseed7' in file:
        #     df_fresh = df_fresh.append(df_temp)
        else:

            df = df.append(df_temp)

# Replace all encoding names with paper friendly versions
for i in range(len(old_names)):
    df = df.replace(old_names[i], order[i])

if args.figure == 'capacity':
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes', order=order)
elif args.figure == 'connected-tiled-maze':
    # 10 seeds each
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']
    # These parameters had the best result on all encodings
    df = df[df['Hidden Layer Size'] == 1024]
    df = df[df['Maze ID Dim'] == 0]
    # for name in order:
    #     print(name, len(df[df['Encoding'] == name]))
    # print(df['Encoding'].unique())
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
elif args.figure == '100-tiled-maze':
    df = df.replace('learned-normalized', 'Learned')
    order = ['Hex SSP', 'RBF', 'Tile-Code', 'Learned']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
elif args.figure == '25-tiled-maze':
    df = df.replace('learned-normalized', 'Learned')
    order = ['Hex SSP', 'RBF', 'Tile-Code', 'Learned']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
elif args.figure == 'tiled-maze':
    # dim is 1024 for these experiments
    # 10 seeds each
    df = df[df['Hidden Layers'] == 1]
    df = df[df['Hidden Layer Size'] == 2048]
    df = df[df['Maze ID Dim'] == 0]
    df = df.replace('learned-normalized', 'Learned')
    order = ['Hex SSP', 'RBF', 'Tile-Code', 'Learned']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes', order=order)
elif args.figure == 'blocksmaze':
    # print(df['Trained On'].unique())
    # print(df['Encoding'].unique())
    # print(df[df['Encoding'] == 'Learned Normalized']['Angular RMSE'])
    # print(len(df[df['Encoding'] == 'Learned Normalized']))
    # print(len(df[df['Encoding'] == 'Hex SSP']))
    # print(len(df[df['Encoding'] == 'Learned']))
    # print(df['Dataset'])
    df = df[df['Encoding'] == 'Learned']
    # df = df[df['Encoding'] != 'Learned Normalized']
    # df = df.replace(np.nan, 'ssp')

    # df_maze = df[df['Dataset'] == 'maze']
    # df_blocks = df[df['Dataset'] == 'blocks']
    # print(len(df_blocks))
    # print(len(df_maze))
    # order = ['Hex SSP', 'Learned']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df_maze, x='Encoding', y='Angular RMSE', hue='Trained On', ax=ax[0])
    # sns.barplot(data=df_blocks, x='Encoding', y='Angular RMSE', hue='Trained On', ax=ax[1])
    # sns.barplot(data=df_maze, x='Trained On', y='Angular RMSE', ax=ax[0], order=['maze', 'blocks'])
    # sns.barplot(data=df_blocks, x='Trained On', y='Angular RMSE', ax=ax[1], order=['maze', 'blocks'])

    sns.barplot(data=df, x='Dataset', y='Angular RMSE', hue='Trained On', order=['maze', 'blocks'], ax=ax)
    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df_fresh, x='Dataset', y='Angular RMSE', hue='Trained On', order=['maze', 'blocks'], ax=ax)
elif args.figure == 'small-env':
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
elif args.figure == 'large-env':
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
elif args.figure == 'network-size':
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Hidden Layers')

    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)

    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Dimensionality')

    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE')

    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Hidden Layers', y='Angular RMSE')

    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Dimensionality', y='Angular RMSE')
elif args.figure == 'hidden-size':
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Encoding')
elif args.figure == 'dim':
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')

sns.despine()
plt.show()
