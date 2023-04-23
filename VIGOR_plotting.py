import matplotlib.pyplot as plt
import numpy as np
import platform
import os
from matplotlib.lines import Line2D
import warnings
from scipy import stats
import dabest
import pandas as pd


# save animal plot as png
def save_sessionplot_as_png(root, animal, session, filename,
                            dpi='figure', transparent=True, background='auto'):
    sessionPath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session
    folderPath = os.path.join(sessionPath, "Figures")
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    filePath = os.path.join(folderPath, filename)
    plt.savefig(filePath, dpi=dpi, transparent=transparent,
                facecolor=background, edgecolor=background)


# plot old boundaries run stay
def plot_peak(ax, data, leftBoundaryPeak, rightBoundaryPeak, kde,
              maxminstep, maxminstep2, xyLabels=["N", "Bins"]):
    if ax is None:
        ax = plt.gca()

    bins = np.arange(120)
    xx = np.linspace(0, 120, 120)
    xline1 = [leftBoundaryPeak, leftBoundaryPeak]
    xline2 = [rightBoundaryPeak, rightBoundaryPeak]
    border = 5
    yline = [0, 0.01]

    if platform.system() == 'Darwin':
        ax.hist(data, normed=True, bins=bins, alpha=0.3,
                orientation='horizontal')  # bugged on linux, working on mac

    # plot kde + boundaries
    ax.plot(kde(xx), xx, color='r')
    ax.plot(yline, xline1, ":", color='k')
    ax.plot(yline, xline2, ":", color='k')
    ax.plot(yline, [xline1[0] + border, xline1[0] + border],
            ":", c='k', alpha=0.5)
    ax.plot(yline, [xline2[0] - border, xline2[0] - border],
            ":", c='k', alpha=0.5)
    # configure plot
    ax.set_xlim(maxminstep[0], maxminstep[1])
    ax.set_ylim(maxminstep2[0], maxminstep2[1])
    ax.set_xlabel(xyLabels[1])
    ax.set_ylabel(xyLabels[0], labelpad=-1)
    ax.spines['top'].set_color("none")
    ax.spines['left'].set_color("none")
    ax.spines['right'].set_color("none")
    ax.axes.get_yaxis().set_visible(False)
    return ax


def plot_BASEtrajectoryV2(ax, time, running_Xs, idle_Xs, lickL, lickR,
                          rewardProbaBlock, blocks, barplotaxes,
                          xyLabels=[" ", " ", " ", " "]):
    if ax is None:
        ax = plt.gca()

    for i in range(0, len(blocks)):
        ax.axvspan(blocks[i][0], blocks[i][1],
                   color='grey', alpha=rewardProbaBlock[i]/250,
                   label="%reward: " + str(rewardProbaBlock[i])
                   if (i == 0 or i == 1) else "")
    ax.plot(time, running_Xs, label="run", color="dodgerblue", linewidth=1)
    ax.plot(time, idle_Xs, label="wait", color="orange", linewidth=1)
    ax.plot(time, [None if x == 0 else x for x in lickL],
            color="b", marker="o", markersize=1)
    ax.plot(time, [None if x == 0 else x for x in lickR],
            color="b", marker="o", markersize=1)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[2])
    ax.set_xlim([barplotaxes[0], barplotaxes[1]])
    ax.set_ylim([barplotaxes[0], barplotaxes[3]+30])
    return ax


# function to plot the tracks of each runs and stays
def plot_tracks(ax, posdataRight, timedataRight, bounds, xylim,
                color, xyLabels=[" ", " ", " ", " "], title=""):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title, pad=50)
    for i, j in zip(posdataRight, timedataRight):
        ax.plot(np.subtract(j, j[0]), i, color=color[0], linewidth=0.3,
                label="Good Item" if i == posdataRight[0] else "")
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    ax.set_xlim([xylim[0], xylim[1]])
    ax.set_ylim([xylim[2], xylim[3]])
    xline1 = [bounds[0], bounds[0]]
    xline2 = [bounds[1], bounds[1]]
    yline = [0, 20]
    ax.plot(yline, xline1, ":", color='k')
    ax.plot(yline, xline2, ":", color='k')
    ax.legend()
    return ax


# function to plot the cumulative distribution of the run speeds and stay times
def cumul_plot(ax, dataRight, dataLeft, maxminstepbin,
               legend, color, xyLabels=["", ""], title=''):
    if ax is None:
        ax = plt.gca()
    custom_legend = [Line2D([0], [0], color=color[0]),
                     Line2D([0], [0], color=color[1])]
    ax.hist(dataRight,
            np.arange(maxminstepbin[0], maxminstepbin[1], maxminstepbin[2]),
            weights=np.ones_like(dataRight)/float(len(dataRight)), color=color[0],
            histtype='step', cumulative=True)
    ax.hist(dataLeft,
            np.arange(maxminstepbin[0], maxminstepbin[1], maxminstepbin[2]),
            weights=np.ones_like(dataLeft)/float(len(dataLeft)), color=color[1],
            histtype='step', cumulative=True)
    ax.set_title(title, pad=50)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    ax.set_xlim([maxminstepbin[0], maxminstepbin[1]])
    ax.set_ylim([maxminstepbin[0], maxminstepbin[2]])

    ax.legend(custom_legend, [legend[0], legend[1]],
              bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              mode="expand", borderaxespad=0., frameon=False)
    return ax


# function to plot the scatter data of run times and stay times
def distribution_plot(ax, dataRight, dataLeft, scatterplotaxes,
                      legend, color, xyLabels=["", "", "", ""], title=''):
    if ax is None:
        ax = plt.gca()
    ax.scatter(np.random.normal(1, 0.05, len(dataRight)), dataRight,
               s=20, color=color[0], marker="$\u25ba$", label=legend[0])
    ax.scatter(np.random.normal(2, 0.05, len(dataLeft)), dataLeft,
               s=20, color=color[1], marker="$\u25c4$", label=legend[1])

    ax.scatter(1.2, np.mean(dataRight), s=25, color=color[0])
    ax.scatter(2.2, np.mean(dataLeft), s=25, color=color[1])
    ax.boxplot(dataRight, positions=[1.35])
    ax.boxplot(dataLeft, positions=[2.35])
    ax.set_xlabel(xyLabels[1])
    ax.set_ylabel(xyLabels[0])
    ax.set_title(title, pad=50)
    ax.set_xlim([scatterplotaxes[0], scatterplotaxes[1]])
    ax.set_ylim([scatterplotaxes[0], scatterplotaxes[2]])
    ax.set_xticks([1, 2])
    ax.set_xticklabels([xyLabels[2], xyLabels[3]])
    ax.legend()
    return ax


# plot rat speed along run
def plot_speed(ax, posdataRight, timedataRight, bounds, xylim,
               xyLabels=[" ", " ", " ", " "], title=''):
    if ax is None:
        ax = plt.gca()
    for i, j in zip(posdataRight, timedataRight):
        time = np.subtract(j, j[0])
        iabs = [abs(ele) for ele in i]
        plt.plot(np.subtract(j, j[0]), iabs, color='g', linewidth=0.3)
        if len(np.where(i == max(i))[0]) == 1:
            maxspeed = max(iabs)
            maxspeedtime = np.where(iabs == maxspeed)[0]
            plt.scatter(time[maxspeedtime], maxspeed, color='darkgreen', s=20)
        else:
            print("Error in plot_speed()")
    ax.set_title(title)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    ax.set_xlim([xylim[0], xylim[1]])
    ax.set_ylim([xylim[2], xylim[3]])
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    xline1 = [bounds[0], bounds[0]]
    xline2 = [bounds[1], bounds[1]]
    yline = [0, 20]
    plt.plot(yline, xline1, ":", color='k')
    plt.plot(yline, xline2, ":", color='k')
    return ax


def plot_figBin(ax, data, rewardProbaBlock, blocks, barplotaxes, stat, color='k',
                xyLabels=[" ", " ", " ", " "], title="", scatter=False):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    if not ax:
        ax = plt.gca()
    for i in range(0, len(blocks)):
        ax.axvspan(blocks[i][0]/60, blocks[i][1]/60, color='grey',
                   alpha=rewardProbaBlock[i]/250,
                   label="%reward: " + str(rewardProbaBlock[i]) if (i == 0 or i == 1) else "")
        if scatter:
            ax.scatter(np.random.normal(((blocks[i][1] + blocks[i][0])/120),
                       1, len(data[i])), data[i], s=5, color=color)

    if stat == "Avg. ":
        ax.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))],
                [np.mean(data[i]) for i in range(0, len(blocks))],
                marker='o', ms=7, color=color)
        if isinstance(data[0], list):
            ax.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))],
                        [np.mean(data[i]) for i in range(0, len(blocks))],
                        yerr=[stats.sem(data[i]) for i in range(0, len(blocks))],
                        fmt='o', color=color, ecolor='black', elinewidth=1, capsize=0)

    elif stat == "Med. ":
        ax.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))],
                [np.median(data[i]) for i in range(0, len(blocks))],
                marker='o', ms=7, color=color)
        if isinstance(data[0], list):
            ax.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))],
                        [np.median(data[i]) for i in range(0, len(blocks))],
                        yerr=[stats.sem(data[i]) for i in range(0, len(blocks))],
                        fmt='o', color=color, ecolor='black', elinewidth=1, capsize=3)

    ax.set_title(title)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(stat + xyLabels[1])
    ax.set_xlim([barplotaxes[0], barplotaxes[1]])
    ax.set_ylim([barplotaxes[2], barplotaxes[3]])
    return ax


# plot block per %reward
def plot_figBinMean(ax, dataLeft, dataRight, color, ylim):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    two_groups_unpaired = dabest.load(pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in {
            "10%\nreward": dataLeft, "90%\nreward": dataRight}.items()])),
            idx=("10%\nreward", "90%\nreward"), resamples=5000)
    two_groups_unpaired.mean_diff.plot(
        ax=ax, raw_marker_size=7, es_marker_size=5,
        custom_palette={"10%\nreward": color, "90%\nreward": color},
        group_summaries='mean_sd',
        group_summary_kwargs={'lw': 3, 'alpha': 0.8},
        reflines_kwargs={'linestyle': 'dashed',
                         'linewidth': 0.8, 'color': 'black'},
        swarm_ylim=(ylim),
        swarm_label="",
        contrast_label="mean difference",
        halfviolin_alpha=0.5,
        violinplot_kwargs={'widths': 0.5})

    # when plot is ax, swarm is ax, contrast is ax.contrast_axes.
    ax.axvspan(-0.5, 0.5, color='grey', alpha=10/250)
    ax.axvspan(0.5, 1.5, color='grey', alpha=90/250)
    return ax
