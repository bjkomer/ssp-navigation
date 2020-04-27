import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

parser = argparse.ArgumentParser('Generate final version of figures for thesis')

parser.add_argument('--figure', type=str, default='capacity',
    choices=[
        'capacity', 'bounds', 'connected-tiled-maze', '100-tiled-maze', '25-tiled-maze', 'tiled-maze'
    ]
)

args = parser.parse_args()

old_names = ['hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot', 'learned', '2d', 'random']
order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']

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

sns.despine()
plt.show()
