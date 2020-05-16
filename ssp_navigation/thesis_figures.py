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
        'blocksmaze', 'small-env', 'large-env', 'network-size', 'hidden-size', 'dim', 'all-enc-256',
        'scale-params', 'lr', 'batch-size'
    ]
)

args = parser.parse_args()

old_names = [
    'hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot',
    'learned', 'learned-normalized', '2d', 'random',
    'frozen-learned', 'frozen-learned-normalized',
    'random-proj', 'ind-ssp', '2d-normalized', 'legendre'
]
if args.figure == 'all-enc-256':
    # shortened names
    order = [
        'Hex SSP', 'SSP', 'RBF', 'TC', 'OH',
        'Learn', 'Learn-N', '2D', 'Rand',
        'Learn-F', 'Learn-F-N',
        'Rand-P', 'Ind SSP', '2D-N', 'LG'
    ]
else:
    order = [
        'Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot',
        'Learned', 'Learned Normalized', '2D', 'Random',
        'Frozen Learned', 'Frozen Learned Normalized',
        'Random Proj', 'Ind SSP', '2D Normalize', 'Legendre'
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
elif args.figure == 'all-enc-256':
    # folders = ['eval_data_tt/med_dim_adam_exps']
    folders = ['eval_data_tt/med_dim_many_seeds']
elif args.figure == 'scale-params':
    folders = ['eval_data_tt/scale_params']
elif args.figure == 'lr':
    folders = ['eval_data_tt/lr_exps']
elif args.figure == 'batch-size':
    folders = ['eval_data/batch_exps']
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
    df_l = df[df['Encoding'] == 'Learned']
    df_s = df[df['Encoding'] == 'Hex SSP']
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

    sns.barplot(data=df_l, x='Dataset', y='Angular RMSE', hue='Trained On', order=['maze', 'blocks'], ax=ax)
    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df_fresh, x='Dataset', y='Angular RMSE', hue='Trained On', order=['maze', 'blocks'], ax=ax)

    df_m = df_l[df_l['Dataset'] == 'maze']
    df_mm = df_m[df_m['Trained On'] == 'maze']
    mm = df_mm['Angular RMSE'].mean()
    mm_sd = df_mm['Angular RMSE'].std()

    df_bm = df_m[df_m['Trained On'] == 'blocks']
    bm = df_bm['Angular RMSE'].mean()
    bm_sd = df_bm['Angular RMSE'].std()

    df_b = df_l[df_l['Dataset'] == 'blocks']
    df_mb = df_b[df_b['Trained On'] == 'maze']
    mb = df_mb['Angular RMSE'].mean()
    mb_sd = df_mb['Angular RMSE'].std()

    df_bb = df_b[df_b['Trained On'] == 'blocks']
    bb = df_bb['Angular RMSE'].mean()
    bb_sd = df_bb['Angular RMSE'].std()

    df_sm = df_s[df_s['Dataset'] == 'maze']
    sm = df_sm['Angular RMSE'].mean()
    sm_sd = df_sm['Angular RMSE'].std()
    df_sb = df_s[df_s['Dataset'] == 'blocks']
    sb = df_sb['Angular RMSE'].mean()
    sb_sd = df_sb['Angular RMSE'].std()

    # printing the table
    print("Training & Testing & RMSE \\\\ \hline")
    print("Blocks & Blocks & {:.3f} ({:.3f} SD) \\\\ \hline".format(bb, bb_sd))
    print("Blocks & Maze & {:.3f} ({:.3f} SD) \\\\ \hline".format(bm, bm_sd))
    print("Maze & Blocks & {:.3f} ({:.3f} SD) \\\\ \hline".format(mb, mb_sd))
    print("Maze & Maze & {:.3f} ({:.3f} SD) \\\\ \hline".format(mm, mm_sd))
    print("SSP & Blocks & {:.3f} ({:.3f} SD) \\\\ \hline".format(sb, sb_sd))
    print("SSP & Maze & {:.3f} ({:.3f} SD) \\\\ \hline".format(sm, sm_sd))

elif args.figure == 'small-env':
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
elif args.figure == 'large-env':
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
elif args.figure == 'network-size':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Hidden Layers')
    sns.despine()

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)

    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Dimensionality')
    sns.despine()

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', ax=ax)
    # sns.lineplot(data=df, x='Hidden Layer Size', y='Angular RMSE', ax=ax)
    # ax.set(xscale='log')
    sns.despine()

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.barplot(data=df, x='Hidden Layers', y='Angular RMSE')
    # sns.lineplot(data=df, x='Hidden Layers', y='Angular RMSE')
    sns.despine()

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.barplot(data=df, x='Dimensionality', y='Angular RMSE')
elif args.figure == 'hidden-size':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Encoding')
    sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE')
    # sns.lineplot(data=df, x='Hidden Layer Size', y='Angular RMSE', ax=ax)
    # ax.set(xscale='log')
elif args.figure == 'dim':
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')
    sns.despine()

    # version with all the dimensionalities that one-hot has
    df = df[df['Dimensionality'] != 8]
    df = df[df['Dimensionality'] != 32]
    df = df[df['Dimensionality'] != 128]
    df = df[df['Dimensionality'] != 512]

    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')
elif args.figure == 'all-enc-256':
    df = df[df['Number of Mazes'] == 25]
    # order = [
    #     'Hex SSP', 'SSP', 'Ind SSP',
    #     'RBF', 'Legendre', 'Tile-Code', 'One-Hot',
    #     'Learned', 'Learned Norm', '2D', '2D Norm',
    #     'Random Proj', 'Random',
    # ]
    # TODO: organize colour palettes based on class of encoding
    order = [
        'Hex SSP', 'SSP', 'Ind SSP',
        'RBF', 'LG',
        'TC', 'OH',
        'Learn', 'Learn-N', '2D', '2D-N',
        'Rand-P', 'Rand',
    ]
    ssp_colour = sns.light_palette("orange")#sns.color_palette(, 3)
    cont_colour = sns.light_palette("blue")
    disc_colour = sns.light_palette("green")
    learn_colour = sns.light_palette("purple")
    rand_colour = sns.light_palette("red")

    colours = [
        ssp_colour[3], ssp_colour[4], ssp_colour[5],
        cont_colour[3], cont_colour[4],
        disc_colour[3], disc_colour[4],
        learn_colour[2], learn_colour[3], learn_colour[4], learn_colour[5],
        rand_colour[3], rand_colour[4],
    ]
    # sns.palplot(colours)
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 4), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order, palette=colours)

    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')
    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df, x='Encoding', y='Train Angular RMSE', hue='Number of Mazes')
elif args.figure == 'scale-params':
    df_rbf = df[df['Encoding'] == 'RBF']
    # print(df_rbf)
    df_ssp= df[df['Encoding'] == 'Hex SSP']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df_rbf, x='Sigma', y='Angular RMSE')
    # sns.barplot(data=df_rbf, x='Angular RMSE', y='Angular RMSE')
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df_ssp, x='SSP Scaling', y='Angular RMSE')

    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df_rbf, x='Sigma', y='Train Angular RMSE')
    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df_ssp, x='SSP Scaling', y='Train Angular RMSE')
elif args.figure == 'lr':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df, x='Learning Rate', y='Angular RMSE')
    sns.lineplot(data=df, x='Learning Rate', y='Angular RMSE', ax=ax)
    ax.set(xscale='log')
elif args.figure == 'batch-size':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df, x='Batch Size', y='Angular RMSE')
    sns.lineplot(data=df, x='Batch Size', y='Angular RMSE', ax=ax)
    ax.set(xscale='log')

sns.despine()
plt.show()
