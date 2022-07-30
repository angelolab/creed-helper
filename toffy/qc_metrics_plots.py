from ark.utils.load_utils import load_imgs_from_tree, load_imgs_from_dir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from toffy import qc_comp
from ark.utils import io_utils
import pandas as pd


def call_violin_swarm_plot(plotting_df, fig_label, figsize=(20, 3), fig_dir=None):
    """Makes violin plot with swarm dots. Used with make_batch_effect_plot()

    Args: pandas df with "sample", "channel", "tma", "99.9th_percentile"
          figsize. this is wide and short
          fig dir to save files to. if none it will still present in a jupyter notebook
    Output: show plots or save them to file

    """
    plt.figure(figsize=figsize)
    ax = sns.violinplot(x="channel", y="99.9th_percentile", data=plotting_df,
                        inner=None, scale="width", color="gray")
    ax = sns.swarmplot(x="channel", y="99.9th_percentile", data=plotting_df,
                       edgecolor="black", hue="tma", palette="tab20")
    ax.set_title(fig_label)
    plt.xticks(rotation=45)
    if fig_dir:
        plt.savefig(fig_dir+fig_label+"_batch_effects.png", dpi=300)
    plt.show()
    plt.close()


def make_batch_effect_plot(data_dir,
                           normal_tissues,
                           exclude_channels=None,
                           img_sub_folder=None,
                           qc_metric="99.9th_percentile",
                           fig_dir=None):
    """Makes violin plots based on tissue type. Points colored by TMA of origin.
    Args:
    normal_tissues is a list of the tissue type substring to match
    exclude channels is a list of channels to not plot
    sub folder in case theres additional sub folder structure
    qc metric is 99.9th percentile but could be changed to anything in the future
    fig dir to save plots in. not required if you want to just show them in a notebook
    """
    for i in range(len(normal_tissues)):
        samples = io_utils.list_folders(dir_name=data_dir, substrs=normal_tissues[i])
        data = load_imgs_from_tree(data_dir=data_dir,
                                   img_sub_folder=img_sub_folder,
                                   fovs=samples)
        channels = list(data.channels.values)
        if exclude_channels:
            channels = [x for x in channels if x not in exclude_channels]

        plotting_df = pd.DataFrame(columns=["sample", "channel", "tma", "99.9th_percentile"])

        for j in range(len(channels)):
            qc_metrics_per_channel = []

            for k in range(len(samples)):
                tma = [x for x in samples[k].split("_") if "TMA" in x][0]
                qc_metrics_per_channel += [[normal_tissues[i],
                                           channels[j],
                                           tma,
                                           qc_comp.compute_99_9_intensity(data.loc[samples[k],
                                                                          :,
                                                                          :,
                                                                          channels[j]])]]

            plotting_df = plotting_df.append(pd.DataFrame(qc_metrics_per_channel,
                                                          columns=plotting_df.columns))

        call_violin_swarm_plot(plotting_df, fig_label=normal_tissues[i], fig_dir=fig_dir)
