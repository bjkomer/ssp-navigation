import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.patches as mpatches
import matplotlib.ticker

parser = argparse.ArgumentParser('Generate final version of figures for thesis')

parser.add_argument('--figure', type=str, default='capacity',
    choices=[
        'capacity', 'bounds', 'connected-tiled-maze', '100-tiled-maze', '25-tiled-maze', 'tiled-maze',
        'blocksmaze', 'small-env', 'large-env', 'network-size', 'hidden-size', 'dim', 'all-enc-256',
        'scale-params', 'lr', 'batch-size', 'proj-exps', 'integ-policy', 'all-enc-256-large-hs',
        'proj-st-exps', 'proj-st-tiled-exps', 'localization', 'large-exp-tiled',
        'hs-enc', 'lr-enc', 'bs-enc', 'nl-enc',
        'learned-phi-reg', 'learned-phi-no-reg',
        'large-dim-adam',
        'learned-phi-reg-v2',
        'connected-tiled-maze-512',
        'connected-tiled-maze-1024',
    ]
)

args = parser.parse_args()

old_names = [
    'hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot',
    'learned', 'learned-normalized', '2d', 'random',
    'frozen-learned', 'frozen-learned-normalized',
    'random-proj', 'ind-ssp', '2d-normalized', 'legendre',
]
if args.figure == 'all-enc-256' or args.figure == 'all-enc-256-large-hs':
    # shortened names
    order = [
        'Hex SSP', 'SSP', 'RBF', 'TC', 'OH',
        'Learn', 'Learn-N', '2D', 'Rand',
        'Learn-F', 'Learn-F-N',
        'Rand-P', 'Ind SSP', '2D-N', 'LG',
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
    # folders = ['eval_data_tt/med_dim_connected_tiledmaze_more_samples_exps']
    folders = ['eval_data_tt/large_dim_connected_tiledmaze_more_samples_exps']
    # folders = ['eval_data_tt/real_final_conn_tiled']
elif args.figure == '100-tiled-maze':
    folders = ['eval_data_tt/large_dim_100tiledmaze_more_samples_exps']
elif args.figure == '25-tiled-maze':
    folders = ['eval_data_tt/large_dim_25tiledmaze_more_samples_exps']
elif args.figure == 'tiled-maze':
    folders = ['eval_data_tt/large_dim_25tiledmaze_more_samples_exps', 'eval_data_tt/large_dim_100tiledmaze_more_samples_exps']
elif args.figure == 'blocksmaze':
    # folders = ['eval_data_tt/final_blocksmaze_exps']
    folders = ['eval_data_tt/final_moredata_blocksmaze_exps']
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
    # folders = ['eval_data_tt/real_final_dim']
elif args.figure == 'all-enc-256':
    # folders = ['eval_data_tt/med_dim_adam_exps']
    folders = ['eval_data_tt/med_dim_many_seeds']
    folders = ['eval_data_tt/real_final_all_enc']
    # folders = ['eval_data_tt/real_final_all_enc_tiled']
elif args.figure == 'all-enc-256-large-hs':
    folders = ['eval_data_tt/final_larger_hidsize_med_dim', 'eval_data_tt/med_dim_many_seeds']
    # folders = ['eval_data_tt/final_larger_hidsize_med_dim']
elif args.figure == 'scale-params':
    folders = ['eval_data_tt/scale_params']
    # folders = ['eval_data_tt/longer_scale_params']
    folders = ['eval_data_tt/longer_tiled_scale_params']
    folders = ['eval_data_tt/real_final_scale_params']
    folders = ['eval_data_tt/real_final_scale_params_tiled']
elif args.figure == 'lr':
    folders = ['eval_data_tt/lr_exps']
elif args.figure == 'batch-size':
    folders = ['eval_data/batch_exps']
elif args.figure == 'proj-exps':
    folders = ['eval_data_tt/ssp_proj_exps']
elif args.figure == 'proj-st-exps':
    folders = ['eval_data_tt/ssp_st_proj_exps']
elif args.figure == 'proj-st-tiled-exps':
    folders = ['eval_data_tt/ssp_st_proj_tiled_exps']
elif args.figure == 'integ-policy':
    folders = ['eval_data_tt/integ_policy_exps', 'eval_data_tt/integ_policy_longer_exps', 'eval_data_tt/integ_policy_much_longer_exps']
    # folders = ['eval_data_tt/integ_policy_much_longer_exps']
elif args.figure == 'localization':
    # folders = ['eval_loc']
    # folders = ['eval_loc_longer']
    folders = ['eval_loc_final']
    # folders = ['eval_loc_final', 'eval_loc_longer']
    folders = ['eval_loc_final_longer']
elif args.figure == 'large-exp-tiled':
    folders = ['eval_data_tt/final_tiled_large_dim_many_seeds']
elif args.figure == 'hs-enc':
    # folders = ['eval_data_tt/hp_exps/hidden_size']
    folders = ['eval_data_tt/hp_exps_longer/hs']
    folders = ['eval_data_tt/hp_exps_two_layer/hs']
    # folders = ['eval_data_tt/hp_exps_reg/hs']
elif args.figure == 'lr-enc':
    # folders = ['eval_data_tt/hp_exps/lr']
    folders = ['eval_data_tt/hp_exps_longer/lr']
    folders = ['eval_data_tt/hp_exps_two_layer/lr']
    # folders = ['eval_data_tt/hp_exps_reg/lr']
elif args.figure == 'bs-enc':
    # folders = ['eval_data_tt/hp_exps/batch_size']
    folders = ['eval_data_tt/hp_exps_longer/bs']
    folders = ['eval_data_tt/hp_exps_two_layer/bs']
    # folders = ['eval_data_tt/hp_exps_reg/bs']
elif args.figure == 'nl-enc':
    # folders = ['eval_data_tt/hp_exps/nlayers']
    folders = ['eval_data_tt/hp_exps_longer/nl']
    folders = ['eval_data_tt/hp_exps_two_layer/nl']
    # folders = ['eval_data_tt/hp_exps_reg/nl']
elif args.figure == 'learned-phi-reg':
    # folders = ['eval_data_tt/learned_phi_regularized']
    folders = ['eval_data_tt/learned_phi_regularized_longer']
    folders = ['eval_data_tt/learned_phi_proper_reg']
    # folders = ['eval_data_tt/learned_phi_large_dim_proper_reg']
    # folders = ['eval_data_tt/learned_phi_proper_reg_phi_decay']
    # folders = ['eval_data_tt/learned_phi_proper_reg_noise']
elif args.figure == 'learned-phi-reg-v2':
    folders = ['eval_data_tt/learned_phi_regularized_longer',
               'eval_data_tt/learned_phi_proper_reg',
               'eval_data_tt/learned_phi_proper_reg_noise'
               ]
    # folders = ['eval_data_tt/learned_phi_regularized_longer',
    #            'eval_data_tt/learned_phi_proper_reg_phi_decay',
    #            'eval_data_tt/learned_phi_proper_reg_noise'
    #            ]
elif args.figure == 'learned-phi-no-reg':
    folders = ['eval_data_tt/learned_phi_no_reg_longer']
elif args.figure == 'large-dim-adam':
    folders = ['eval_data_tt/large_dim_adam_exps']
elif args.figure == 'connected-tiled-maze-512':
    folders = ['eval_data_tt/large_dim_connected_reg_v2']
elif args.figure == 'connected-tiled-maze-1024':
    folders = ['eval_data_tt/large_dim_connected_reg']
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
    # df = df.replace('learned-normalized', 'Learned')
    df = df.replace('Learned Normalized', 'Learned')
    order = ['Hex SSP', 'RBF', 'Tile-Code', 'Learned']
    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)

    # for cogsci version of the figure
    fix, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)
    label_fontsize = 15  # 20
    tick_fontsize = 12  # 16
    ax.set_xlabel('Encoding', fontsize=label_fontsize)
    ax.set_ylabel('RMSE', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
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
    df = df[df['Encoding'] == 'SSP']
    # df = df[df['Encoding'] == 'Hex SSP']
    # df = df[df['Encoding'] == 'RBF']
    # df = df[df['Encoding'] == 'Tile-Code']
    # df = df[df['Encoding'] == 'One-Hot']
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
    # df = df[df['Seed'] >= 8]
    # df = df[df['Hidden Layer Size'] == 256]
    # df = df[df['Hidden Layer Size'] == 1024]
    # df = df[df['Hidden Layer Size'] == 2048]
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

    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df_rbf, x='Sigma', y='Train Angular RMSE')
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df_ssp, x='SSP Scaling', y='Train Angular RMSE')

    fix, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True, sharey=True)
    sns.lineplot(data=df_ssp, x='SSP Scaling', y='Train Angular RMSE', ax=ax[0])
    sns.lineplot(data=df_rbf, x='Sigma', y='Train Angular RMSE', ax=ax[1])

    fix, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True, sharey=True)
    sns.lineplot(data=df_ssp, x='SSP Scaling', y='Angular RMSE', ax=ax[0])
    sns.lineplot(data=df_rbf, x='Sigma', y='Angular RMSE', ax=ax[1])

    # fix, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True, sharey=True)
    # sns.lineplot(data=df_ssp, x='SSP Scaling', y='Train Loss', ax=ax[0])
    # sns.lineplot(data=df_rbf, x='Sigma', y='Train Loss', ax=ax[1])
    #
    # fix, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True, sharey=True)
    # sns.lineplot(data=df_ssp, x='SSP Scaling', y='Test Loss', ax=ax[0])
    # sns.lineplot(data=df_rbf, x='Sigma', y='Test Loss', ax=ax[1])

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
elif args.figure == 'proj-exps':
    df_st = df[df['Encoding'] == 'sub-toroid-ssp']
    df_var_st = df[df['Encoding'] == 'var-sub-toroid-ssp']
    df_proj = df[df['Encoding'] == 'proj-ssp']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df_st, x='Scale Ratio', y='Angular RMSE', hue='Proj Dim')
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df_proj, x='Encoding', y='Angular RMSE', hue='Proj Dim')
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df_var_st, x='Encoding', y='Angular RMSE')
elif args.figure == 'proj-st-exps' or args.figure == 'proj-st-tiled-exps':
    df = df[df['Proj Dim'] != 50]
    df = df.rename(columns={'Proj Dim': 'Sub-Toroid Dimension'})
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Sub-Toroid Dimension', y='Angular RMSE', palette='hls')
    ax.set_ylim([0.35, 0.42])
elif args.figure == 'integ-policy':
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Epochs')
    df = df[df['Epochs'] >= 100]
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')
    df = df[df['Number of Mazes'] == 5]
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Hidden Layer Size')
    df = df[df['Hidden Layer Size'] >= 2048]
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Dimensionality')
elif args.figure == 'all-enc-256-large-hs':
    # df = df[df['Number of Mazes'] == 10]
    df = df[df['Number of Mazes'] == 25]
    # df = df[df['Number of Mazes'] == 50]
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

    df_256 = df[df['Hidden Layer Size'] == 256]
    df_512 = df[df['Hidden Layer Size'] == 512]
    df_1024 = df[df['Hidden Layer Size'] == 1024]

    # sns.palplot(colours)
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 4), tight_layout=True)
    sns.barplot(data=df_256, x='Encoding', y='Angular RMSE', order=order, palette=colours)
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 4), tight_layout=True)
    sns.barplot(data=df_512, x='Encoding', y='Angular RMSE', order=order, palette=colours)
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 4), tight_layout=True)
    sns.barplot(data=df_1024, x='Encoding', y='Angular RMSE', order=order, palette=colours, ax=ax)

    test_results = add_stat_annotation(
        ax, data=df_1024, x='Encoding', y='Angular RMSE', order=order,
        # box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
        # test='Mann-Whitney',
        comparisons_correction=None,
        box_pairs=[("Ind SSP", "RBF"), ("Hex SSP", "RBF"), ("Hex SSP", "SSP"), ("SSP", "RBF"), ("SSP", "Ind SSP") ],
        test='t-test_ind',
        text_format='star',
        loc='inside',
        verbose=2
    )


    # Print mean and SD for each encoding
    for enc in order:
        mean = df_1024[df_1024['Encoding'] == enc]['Angular RMSE'].mean()
        std = df_1024[df_1024['Encoding'] == enc]['Angular RMSE'].std()
        print(enc)
        print("Mean: {}".format(mean))
        print("STD: {}".format(std))
        print("")

    # calculate p-values between top encodings with student t-test
    data_hex_ssp = df_1024[df_1024['Encoding'] == 'Hex SSP']['Angular RMSE']
    data_ssp = df_1024[df_1024['Encoding'] == 'SSP']['Angular RMSE']
    data_ind_ssp = df_1024[df_1024['Encoding'] == 'Ind SSP']['Angular RMSE']
    data_rbf = df_1024[df_1024['Encoding'] == 'RBF']['Angular RMSE']

    print("Hex SSP to SSP")
    stat, p = ttest_ind(data_hex_ssp, data_ssp)
    print('stat=%.3f, p=%.5f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    print("Hex SSP to RBF")
    stat, p = ttest_ind(data_hex_ssp, data_rbf)
    print('stat=%.3f, p=%.5f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    print("Ind SSP to RBF")
    stat, p = ttest_ind(data_ind_ssp, data_rbf)
    print('stat=%.3f, p=%.5f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    print("SSP to RBF")
    stat, p = ttest_ind(data_ssp, data_rbf)
    print('stat=%.3f, p=%.5f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    print("SSP to Ind SSP")
    stat, p = ttest_ind(data_ssp, data_ind_ssp)
    print('stat=%.3f, p=%.5f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

elif args.figure == 'localization':
    # order = ['sub-toroid-ssp', 'Hex SSP', 'SSP', 'RBF', 'Legendre', 'Tile-Code', 'One-Hot', '2D', '2D Normalize', 'Random']
    # order = ['sub-toroid-ssp', 'Hex SSP', 'SSP', 'RBF', 'Legendre', 'Tile-Code', 'One-Hot', '2D']
    order = [
        'sub-toroid-ssp',
        'SSP', 'Hex SSP',
        'RBF', 'Legendre',
        'One-Hot', 'Tile-Code',
        '2D',# 'Random',
    ]
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 4), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='RMSE', order=order)
    # fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    # sns.barplot(data=df, x='Encoding', y='MSE Loss', order=order)

    test_results = add_stat_annotation(
        ax, data=df, x='Encoding', y='RMSE', order=order,
        # box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
        # test='Mann-Whitney',
        comparisons_correction=None,
        box_pairs=[
            ("RBF", "Legendre"),
            ("SSP", "Hex SSP"),
            ("Hex SSP", "RBF"),
            ("Legendre", "2D"),
            ("SSP", "Legendre"),
            ("SSP", "RBF"),
            ("Hex SSP", "Legendre"),
            ("SSP", "2D"),
        ],
        test='t-test_ind',
        text_format='star',
        loc='inside',
        verbose=2
    )

elif args.figure == 'large-exp-tiled':
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE')

    df = df[df['Encoding'] == 'RBF']
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Hilbert Curve', y='Angular RMSE')
elif args.figure == 'hs-enc':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Encoding', ax=ax)
    sns.lineplot(data=df, x='Hidden Layer Size', y='Angular RMSE', hue='Encoding', ax=ax)
    ax.set(xscale='log')

    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.lineplot(data=df, x='Hidden Layer Size', y='Angular RMSE', ax=ax)

    # ax.set(xscale='log')
    ax.set_xscale('log')

    # ax.set_xticks([])

    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([256, 512, 1024, 2048])
    ax.set_xticklabels([256, 512, 1024, 2048])
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    # consistent y-limit across plots
    ax.set_ylim([.35, .6])
elif args.figure == 'lr-enc':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df, x='Learning Rate', y='Angular RMSE', hue='Encoding', ax=ax)
    sns.lineplot(data=df, x='Learning Rate', y='Angular RMSE', hue='Encoding', ax=ax)
    ax.set(xscale='log')

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.lineplot(data=df, x='Learning Rate', y='Angular RMSE', ax=ax)
    ax.set(xscale='log')
    # consistent y-limit across plots
    ax.set_ylim([.35, .6])
elif args.figure == 'bs-enc':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df, x='Batch Size', y='Angular RMSE', hue='Encoding', ax=ax)
    sns.lineplot(data=df, x='Batch Size', y='Angular RMSE', hue='Encoding', ax=ax)
    ax.set(xscale='log')

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.lineplot(data=df, x='Batch Size', y='Angular RMSE', ax=ax)
    ax.set(xscale='log')
    # consistent y-limit across plots
    ax.set_ylim([.35, .6])
elif args.figure == 'nl-enc':
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df, x='Hidden Layers', y='Angular RMSE', hue='Encoding', ax=ax)
    sns.lineplot(data=df, x='Hidden Layers', y='Angular RMSE', hue='Encoding', ax=ax)

    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.lineplot(data=df, x='Hidden Layers', y='Angular RMSE', ax=ax)
    ax.set_xticks([1, 2, 3, 4])
    # consistent y-limit across plots
    ax.set_ylim([.35, .6])
elif args.figure == 'learned-phi-reg' or args.figure == 'learned-phi-no-reg':
    # df_256 = df[df['Hidden Layer Size'] == 256]
    df_256 = df
    # df_1024 = df[df['Hidden Layer Size'] == 1024]
    fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    sns.barplot(data=df_256, x='Encoding', y='Angular RMSE', ax=ax)
    # fix, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # sns.barplot(data=df_1024, x='Encoding', y='Angular RMSE', ax=ax)

    old_names = [
        'learned-ssp', 'sub-toroid-ssp', 'SSP', 'Learned',
    ]

    order = [
        'Learned SSP', 'Fixed Grid SSP', 'Fixed SSP', 'Learned Encoding',
    ]

    for i in range(len(old_names)):
        df_256 = df_256.replace(old_names[i], order[i])

    fix, ax = plt.subplots(1, 1, figsize=(6, 3), tight_layout=True)
    sns.barplot(data=df_256, x='Encoding', y='Angular RMSE', ax=ax, order=order)

    test_results = add_stat_annotation(
        ax, data=df_256, x='Encoding', y='Angular RMSE', order=order,
        comparisons_correction=None,
        box_pairs=[
            (order[0], order[1]),
            # (order[2], order[3]),
            (order[0], order[2]),
            (order[0], order[3]),
            # (order[1], order[2]),
            # (order[1], order[3]),
        ],
        test='t-test_ind',
        text_format='star',
        loc='inside',
        verbose=2
    )
    sns.despine()

    df_256_train = df_256.copy()
    df_256_train = df_256_train.drop(columns=['Angular RMSE'])
    df_256_train = df_256_train.rename(columns={"Train Angular RMSE": "Angular RMSE"})

    df_256_train['Dataset'] = 'Train'
    df_256['Dataset'] = 'Test'

    df_compare = pd.concat([df_256_train, df_256])

    # palette_train = sns.color_palette('muted')
    # palette_test = sns.color_palette('deep')

    fix, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    bar = sns.barplot(
        data=df_compare, x='Dataset', y='Angular RMSE', hue='Encoding', ax=ax, order=['Train', 'Test'], hue_order=order,
    )

    if args.figure == 'learned-phi-reg':
        ax.get_legend().set_visible(False)
    elif args.figure == 'learned-phi-no-reg':
        ax.set_xlabel('')

    # fix, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
    # bar = sns.barplot(
    #     data=df_compare, x='Encoding', y='Angular RMSE', hue='Dataset', ax=ax, order=order, hue_order=['Train', 'Test'],
    #     # color=list(palette_train[:4]) + list(palette_test[:4])
    #     # palette=palette_test
    # )

    # colors = ["red", "green", "blue", "black"]
    # # Loop over the bars
    # for i, thisbar in enumerate(bar.patches):
    #     # Set a different hatch for each bar
    #     thisbar.set_color(colors[i])
    #     thisbar.set_edgecolor("white")

    # test_results = add_stat_annotation(
    #     ax, data=df_compare, x='Encoding', y='Angular RMSE', order=order,
    #     comparisons_correction=None,
    #     box_pairs=[
    #         (order[0], order[1]),
    #         # (order[2], order[3]),
    #         (order[0], order[2]),
    #         (order[0], order[3]),
    #         # (order[1], order[2]),
    #         # (order[1], order[3]),
    #     ],
    #     test='t-test_ind',
    #     text_format='star',
    #     loc='inside',
    #     verbose=2
    # )
elif args.figure == 'learned-phi-reg-v2':
    df_original = df[df['Batch Size'] == 64]
    df_original = df_original[df_original['Hidden Layer Size'] == 1024]
    df_new = df[df['Batch Size'] == 512]
    df_new_noise = df_new[df_new['Input Noise'] > 0]
    df_new_no_noise = df_new[df_new['Input Noise'] == 0]

    df_original['Version'] = 'Old'
    df_new_no_noise['Version'] = 'No Noise'
    df_new_noise['Version'] = 'Noise'

    df_combined = pd.concat([df_original, df_new_no_noise, df_new_noise])
    fix, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    bar = sns.barplot(
        data=df_combined, x='Version', y='Angular RMSE',
    )

elif args.figure == 'large-dim-adam':
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes')
elif args.figure == 'connected-tiled-maze-512' or args.figure == 'connected-tiled-maze-1024':
    df = df.replace('sub-toroid-ssp', 'Grid SSP')
    order = ['Grid SSP', 'Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'Learned Normalized', '2D Normalize']
    df = df[df['Hidden Layer Size'] == 2048]
    df = df[df['Maze ID Dim'] == 0]
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)

sns.despine()
plt.show()
