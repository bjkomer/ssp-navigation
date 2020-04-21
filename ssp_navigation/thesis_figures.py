import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

parser = argparse.ArgumentParser('Generate final version of figures for thesis')

parser.add_argument('--figure', type=str, default='capacity',
                    choices=['capacity', 'bounds'])

args = parser.parse_args()

old_names = ['hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot', 'learned', '2d', 'random']
order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']

if args.figure == 'capacity':
    folder = 'eval_data_tt/final_capacity_exps/'

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

df = pd.DataFrame()

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
    fix, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), tight_layout=True)
    sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes', order=order)

sns.despine()
plt.show()
