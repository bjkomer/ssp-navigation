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

if os.path.isfile(combined_fname):
    # combined file already exists, load it
    df = pd.read_csv(combined_fname)
else:
    # combined file does not exist, create it
    files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]

    df = pd.DataFrame()

    for file in files:
        df_temp = pd.read_csv(os.path.join(args.folder, file))

        df.append(df_temp)

    df.to_csv(combined_fname)

sns.barplot(data=df, x='Encoding', y='RMSE')
plt.show()
