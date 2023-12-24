import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def gather_dataset_stats(cfg):
    root_path = os.path.abspath(os.path.join(cfg.data.train, '..', '..'))
    class_idx = {
        str(i): cfg.data.names[i]
        for i in range(cfg.data.nc)
    }

    class_stat = {}
    data_len = {}
    class_info = []

    for mode in ['train', 'valid', 'test']:
        class_count = {
            cfg.data.names[i]: 0
            for i in range(cfg.data.nc)
        }

        label_path = os.path.join(root_path, mode, 'labels')

        for file in os.listdir(label_path):
            with open(os.path.join(label_path, file)) as f:
                lines = f.readlines()

                for cls in set([line[0] for line in lines]):
                    class_count[class_idx[cls]] += 1

        data_len[mode] = len(os.listdir(label_path))
        class_stat[mode] = class_count

        class_info.append({
            'Mode': mode,
            **class_count,
            'Data_Volume': data_len[mode]
        })

    dataset_stats_df = pd.DataFrame(class_info)
    return dataset_stats_df


def plot_dataset_stats(dataset_stats_df):
    # Create subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot vertical bar plots for each mode in subplots
    for i, mode in enumerate(['train', 'valid', 'test']):
        sns.barplot(
            data=dataset_stats_df[dataset_stats_df['Mode'] == mode].drop(columns='Mode'),
            orient='v',
            ax=axes[i],
            palette='Set2'
        )

        axes[i].set_title(f'{mode.capitalize()} Class Statistics')
        axes[i].set_xlabel('Classes')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=90)

        # Add annotations on top of each bar
        for p in axes[i].patches:
            axes[i].annotate(
                f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                textcoords='offset points'
            )

    plt.tight_layout()
    plt.show()
