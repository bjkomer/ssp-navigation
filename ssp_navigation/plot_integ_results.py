import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

single_result = False

blocks_indices = [0, 4, 6, 7, 8, 9]
maze_indices = [1, 2, 3, 5]


if single_result:
    fname = 'output/results_integ_noise0.25.npz'

    returns = np.load(fname)['returns']

    n_mazes = returns.shape[0]
    n_episodes = returns.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)

    # ax.plot(returns[0, :])
    # ax.plot(returns[1, :])
    # ax.plot(returns.mean(axis=1))
    ax.scatter(np.arange(n_mazes), returns.mean(axis=1))
else:
    fnames = [
        'output/results_integ_noise0.25_s13_50envs.npz',
        # 'output/results_integ_noise0.25_s13.npz',
        'output/results_integ_noise0.25_s13_50envs_loc_gt.npz',
        # 'output/results_integ_noise0.25_s13_loc_gt.npz',
        'output/results_integ_noise0.25_s13_50envs_cleanup_gt.npz',
        # 'output/results_integ_noise0.25_s13_cleanup_gt.npz',
        'output/results_integ_noise0.25_s13_50envs_cleanup_gt_loc_gt.npz',
        # 'output/results_integ_noise0.25_s13_cleanup_gt_loc_gt.npz',
    ]

    # for the overall percent plot with the four conditions
    df = pd.DataFrame()

    conditions = [
        'NN-Cleanup\nNN-Localization',
        'NN-Cleanup\nGT-Localization',
        'GT-Cleanup\nNN-Localization',
        'GT-Cleanup\nGT-Localization',
    ]

    # Average return on each environment
    fig, ax = plt.subplots(2, 2, figsize=(8, 4), tight_layout=True)

    # Percent of successful trials
    fig_per, ax_per = plt.subplots(2, 2, figsize=(8, 4), tight_layout=True)
    n_maze_max = 10
    # per_condition = np.zeros((4, ))
    for i, fname in enumerate(fnames):
        returns = np.load(fname)['returns']#[:10, :]
        # print(returns.shape)
        n_mazes = returns.shape[0]
        # n_mazes = 10
        n_episodes = returns.shape[1]

        ax[i//2, i % 2].scatter(np.arange(n_mazes), returns.mean(axis=1))
        ax[i // 2, i % 2].set_ylim([-1, 1])
        ax[i // 2, i % 2].set_title(fname)

        per_trials = np.zeros((n_maze_max,))
        for n in range(n_mazes):
            for e in range(n_episodes):
                if returns[n, e] > -1:
                    per_trials[n % n_maze_max] += 1./n_episodes * (n_maze_max / n_mazes)

        for n in range(n_maze_max):
            df = df.append(
                {
                    'Maze Index': n,
                    'Success Rate': per_trials[n],
                    'Condition': conditions[i],
                },
                ignore_index=True,
            )

        # per_trials = np.zeros((n_mazes,))
        # for n in range(n_mazes):
        #     for e in range(n_episodes):
        #         if returns[n, e] > -1:
        #             per_trials[n] += 1./n_episodes
        #
        #     df = df.append(
        #         {
        #             'Maze Index': n % n_maze_max,
        #             'Success Rate': per_trials[n],
        #             'Condition': conditions[i],
        #         },
        #         ignore_index=True,
        #     )

        ax_per[i // 2, i % 2].scatter(np.arange(n_maze_max), per_trials)
        ax_per[i // 2, i % 2].set_ylim([0, 1])
        ax_per[i // 2, i % 2].set_title(fname)

        # per_condition[i] = per_trials.mean()

    # save so a figure combining with spiking results can be made
    df.to_csv('integ_results.csv')

    fig_overall, ax_overall = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

    # sns.barplot(data=df, x='Condition', y='Success Rate', ax=ax_overall)
    sns.boxplot(data=df, x='Condition', y='Success Rate', ax=ax_overall)
    sns.despine()

    for i in range(4):
        df_tmp = df[df['Condition'] == conditions[i]]
        print(conditions[i])
        print("mean")
        print(df_tmp['Success Rate'].mean())
        print("std")
        print(df_tmp['Success Rate'].std())
        print("blocks mean")
        print(df_tmp[df_tmp['Maze Index'].isin(blocks_indices)]['Success Rate'].mean())
        print("maze mean")
        print(df_tmp[df_tmp['Maze Index'].isin(maze_indices)]['Success Rate'].mean())
        print("")


plt.show()
