import matplotlib.pyplot as plt
import numpy as np
import platform
import os
from matplotlib.lines import Line2D
import warnings
from scipy import stats
import dabest
import pandas as pd
from sklearn.neighbors import KernelDensity

from VIGOR_utils import *
from VIGOR_MODELS_Functions import *

animalList = ['RatF00', 'RatF01', 'RatF02', 'RatM00', 'RatM01', 'RatM02', 
            'RatF32', 'RatF33', 'RatM31', 'RatM32', 'RatF42', 'RatM40', 'RatM43', 'RatM53', 'RatM54']


def permutation_test_distances(var, shifty=0, h='top', dhs=[0.05, 0.2], barh=.05, ax=None, num_permutations=10000):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    data = np.array([[var[animal][cond] for cond in ['60', '90', '120']] for animal in animalList])
    p_value_60_90 = exact_mc_perm_paired_test(data[:, 0], data[:, 1])
    p_value_60_120 = exact_mc_perm_paired_test(data[:, 0], data[:, 2])
    p_value_90_120 = exact_mc_perm_paired_test(data[:, 1], data[:, 2])

    # p_value_60_90 = stats.wilcoxon(data[:, 0], data[:, 1])[1]
    # p_value_60_120 = stats.wilcoxon(data[:, 0], data[:, 2])[1]
    # p_value_90_120 = stats.wilcoxon(data[:, 1], data[:, 2])[1]



    print(f'p_value_60_90: {p_value_60_90}', f'p_value_60_120: {p_value_60_120}', f'p_value_90_120: {p_value_90_120}')
    if h is 'bottom':
        h = np.min([np.min(data[:, 0]), np.min(data[:, 1]), np.min(data[:, 2])])
        h1, h2, h3 = [h, h, h]
    elif h is 'top':
        h = np.max([np.max(data[:, 0]), np.max(data[:, 1]), np.max(data[:, 2])])
        h1, h2, h3 = [h, h, h]
    else:
        h1, h2, h3 = h
    barplot_annotate_brackets(ax, 0, 1, stars(p_value_60_90), [0, 1, 2], [h1, h1, h1], dh=dhs[0]+shifty, barh=barh, maxasterix=None)
    barplot_annotate_brackets(ax, 0, 2, stars(p_value_60_120), [0, 1, 2], [h2, h2, h2], dh=dhs[1]+shifty, barh=barh, maxasterix=None)
    barplot_annotate_brackets(ax, 1, 2, stars(p_value_90_120), [0, 1, 2], [h3, h3, h3], dh=dhs[0]+shifty, barh=barh, maxasterix=None)


def permutation_test_vbelt(var, shifty=0, h='top', dhs=[0.05, 0.2], xs=[0, 2, 4], barh=.05, ax=None, num_permutations=10000):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    data = np.array([[var[animal][cond] for cond in ['20', '2', 'rev20']] for animal in animalList])
    p_value_20_2 = exact_mc_perm_paired_test(data[:, 0], data[:, 1])
    p_value_2_rev20 = exact_mc_perm_paired_test(data[:, 1], data[:, 2])
    p_value_20_rev20 = exact_mc_perm_paired_test(data[:, 0], data[:, 2])

    # p_value_20_2 = stats.wilcoxon(data[:, 0], data[:, 1])[1]
    # p_value_2_rev20 = stats.wilcoxon(data[:, 1], data[:, 2])[1]
    # p_value_20_rev20 = stats.wilcoxon(data[:, 0], data[:, 2])[1]

    print(f'p_value_20_0: {p_value_20_2}', f'p_value_0_rev20: {p_value_2_rev20}', f'p_value_20_rev20: {p_value_20_rev20}')
    if h is None:
        h = np.min([np.min(data[:, 0]), np.min(data[:, 1]), np.min(data[:, 2])])
        h1, h2, h3 = [h, h, h]
    elif h is 'top':
        h = np.max([np.max(data[:, 0]), np.max(data[:, 1]), np.max(data[:, 2])])
        h1, h2, h3 = [h, h, h]
    else:
        h1, h2, h3 = h
    barplot_annotate_brackets(ax, 0, 1, stars(p_value_20_2), xs, [h1, h1, h1], dh=dhs[0]+shifty, barh=barh, maxasterix=None)
    barplot_annotate_brackets(ax, 1, 2, stars(p_value_2_rev20), xs, [h2, h2, h2], dh=dhs[0]+shifty, barh=barh, maxasterix=None)
    barplot_annotate_brackets(ax, 0, 2, stars(p_value_20_rev20), xs, [h3, h3, h3], dh=dhs[1]+shifty, barh=barh, maxasterix=None)


def letter_on_subplot(ax, letter, x_rel=-0.2, y_rel=1.15, fs=7):
    ax.annotate(letter, xy=(x_rel, y_rel), xycoords='axes fraction',
                fontsize=fs, fontweight='bold', va='top')

def zoomingBox(fig, main_ax, roi, zoom_ax, color='k', linewidth=.5, roiKwargs={}, arrowKwargs={}, dstForced=None):
    '''
    Create a zooming effect between two subplots using a zooming rectangle.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure containing the subplots.
    - main_ax (matplotlib.axes._axes.Axes): The main subplot where the zooming rectangle will be drawn.
    - roi (list): The coordinates of the zooming rectangle in the form [x_min, x_max, y_min, y_max].
    - zoom_ax (matplotlib.axes._axes.Axes): The subplot where the zoomed-in content will be displayed.
    - color (str, optional): The color of the zooming rectangle and arrows.
    - linewidth (float, optional): The linewidth of the zooming rectangle and arrows. 
    - roiKwargs (dict, optional): Additional keyword arguments for customizing the zooming rectangle.
    - arrowKwargs (dict, optional): Additional keyword arguments for customizing the arrows.
    - dstForced (list, optional): A list containing the forced destination coordinates for the arrows.

    Returns:
    None

    Example:
    fig, axs = plt.subplots(2, 2)
    axs[1,1].plot(np.random.rand(100))
    zoomingBox(fig=fig, main_ax=axs[1,1], roi=[40,60,0.1,0.9], zoom_ax=axs[0,0], dstForced=[[0.038, 0.551], [0.479, 0.983]])
    '''

    # that's a hack 
    fig.canvas.draw()
    bounds = [_ax.get_position().bounds for _ax in fig.axes]



    main_ax.plot([roi[0],roi[1],roi[1],roi[0],roi[0]], [roi[2],roi[2],roi[3],roi[3],roi[2]], lw=linewidth, ls='--', color=color)
    # zoom_ax.plot([roi[0],roi[1],roi[1],roi[0],roi[0]], [roi[2],roi[2],roi[3],roi[3],roi[2]], lw=linewidth, ls='--', color=color)
    srcCorners = [[roi[0],roi[2]], [roi[0],roi[3]], [roi[1],roi[2]], [roi[1],roi[3]]]
    dstCorners = zoom_ax.get_position().corners()

    srcBB = main_ax.get_position()
    dstBB = zoom_ax.get_position()
    if (dstBB.min[0]>srcBB.max[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.max[0]<srcBB.min[0] and dstBB.min[1]>srcBB.max[1]):
        src = [0, 3]; dst = [0, 3]
    elif (dstBB.max[0]<srcBB.min[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.min[0]>srcBB.max[0] and dstBB.min[1]>srcBB.max[1]):
        src = [1, 2]; dst = [1, 2]
    elif dstBB.max[1] < srcBB.min[1]:
        src = [0, 2]; dst = [1, 3]
    elif dstBB.min[1] > srcBB.max[1]:
        src = [1, 3]; dst = [0, 2]
    elif dstBB.max[0] < srcBB.min[0]:
        src = [0, 1]; dst = [2, 3]
    elif dstBB.min[0] > srcBB.max[0]:
        src = [2, 3]; dst = [0, 1]

    arrowKwargs = dict([('arrowstyle','-'), ('linestyle', 'dashed'), ('color',color), ('linewidth',linewidth), ('shrinkA',0), ('shrinkB',0)] + list(arrowKwargs.items()))
    
    if dstForced is not None:
        for k in range(2):
            main_ax.annotate('', xy=dstForced[k], xytext=srcCorners[src[k]], xycoords='figure fraction', textcoords='data', arrowprops=arrowKwargs)
    else:
        for k in range(2):
            main_ax.annotate('', xy=dstCorners[dst[k]], xytext=srcCorners[src[k]], xycoords='figure fraction', textcoords='data', arrowprops=arrowKwargs)

    # that's a hack
    for i, ax in enumerate(fig.axes):
        ax.set_position(bounds[i])
    fig.canvas.draw()


def plot_colorbar(ax_input=None, x=0, y=0, width=.1, height=.1,
                  label='label', y_label=1.35, labelpad=-17, show_zero=None,
                  txt=True, cmap='autumn', labels=['Low', '0', 'High']):

    if ax_input is None:
        fig, ax_input = plt.subplots(figsize=(0.5, 0.5))

    fig = ax_input.get_figure()
    position = ax_input.get_position()
    cax = ax_input.figure.add_axes([position.x0 + x * position.width,
                              position.y0 + y * position.height,
                              width * position.width,
                              height * position.height])

    cax.xaxis.set_visible(False)
    cax.spines['left'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['top'].set_visible(False)
    cax.spines['right'].set_visible(False)

    N = 4
    c = np.arange(1, 100*N + 1)
    cmap_ = plt.get_cmap(cmap, 100*N)
    dummy_ax = cax.scatter(c, c, c=c, cmap=cmap_)
    cax.cla()


    shift = 50
    if show_zero is not None:
        cb=fig.colorbar(dummy_ax, ax=ax_input, cax=cax, ticks=[np.min(c)+shift, show_zero, np.max(c)-shift])
        cb.ax.set_yticklabels([labels[0], labels[1], labels[2]], rotation=0, fontsize=5)
        cax.axhline(show_zero, color='k', lw=0.5, ls='--')

    else:
        cb=fig.colorbar(dummy_ax, ax=ax_input, cax=cax, ticks=[np.min(c)+shift, np.max(c)-shift])
        cb.ax.set_yticklabels([labels[0], labels[2]], rotation=0, fontsize=5)

    if not txt:
        cb.ax.set_yticklabels([])


    cb.outline.set_edgecolor('k')
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, labelpad=labelpad, y=y_label, rotation=0, fontsize=7)
    cb.ax.yaxis.set_tick_params(size=0)

def space_axes(ax=None, x_ratio_left=1/30, x_ratio_right=1/30, y_ratio=1/30, top_y=0):
    if ax is None:
        fig, ax = plt.subplots()

    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    ax.set_xlim(min_x - x_ratio_left * abs(max_x - min_x), max_x + x_ratio_right * abs(max_x - min_x))
    if min_x > max_x:  # if x axis is reversed
        ax.set_xlim(min_x + x_ratio_left * abs(max_x - min_x), max_x - x_ratio_right * abs(max_x - min_x))
    ax.set_ylim(min_y - y_ratio * abs(max_y - min_y), max_y + top_y * abs(max_y - min_y))

    ax.spines['left'].set_bounds(min_y, max_y)
    ax.spines['bottom'].set_bounds(min_x, max_x)


def annotation_d_vbelt(ax=None, miny=0):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    y = np.diff(sorted([ax.get_ylim()[0], miny]))
    dy = miny - y 

    line_y = (miny-dy) * 0.8
    ax.plot([0, 2], [miny-line_y, miny-line_y], color='k', lw=.75, zorder=1)
    ax.plot([3, 7], [miny-line_y, miny-line_y], color='k', lw=.75, zorder=1)

    text_y = (miny-dy) * 0.3
    ax.annotate(r'$d$' + ' (cm)', xy=(1, miny-text_y), xytext=(1, miny-text_y), ha='center', va='center', fontsize=6)
    ax.annotate(r'$v_{\mathrm{belt}}$' + ' (cm/s)', xy=(5, miny-text_y), xytext=(5, miny-text_y), ha='center', va='center', fontsize=6)


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


def plot_shuffling(shuffles=np.random.randn(1000), observed=1, ylim=[-3, 3], xlim=[0, .5], ylabel=' ', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    # half violin
    violin_parts = ax.violinplot(shuffles, positions=[0], showextrema=False, widths=0.5)
    for pc in violin_parts['bodies']:

        pc.set_facecolor('gray')
        pc.set_edgecolor('gray')
        pc.set_alpha(0.25)
        pc.set_linewidth(0)

    # observed
    ax.axhline(observed, 0, .75, color='gray')

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    space_axes(ax, x_ratio_left=.0)


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


# group bin data by reward%
def poolByReward(data, proba, blocks, rewardproba):
    output = []
    for i in range(0, len(blocks)):
        if rewardproba[i] == proba:
            if len(data) == 1:
                output.append(data[0][i])
            if len(data) == 2:  # usually for data like dataLeft+dataRight
                output.append(data[0][i]+data[1][i])
            if len(data) > 2:
                print("too much data, not intended")
    return output


# separate data by condition
def separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, bin):

    def fix_singleRun(data):
        recoveredRun = []
        indexList = []
        fixedList = data.copy()
        for index, ele in enumerate(data):
            if not isinstance(ele, list):
                recoveredRun.append(np.float64(ele))
                indexList.append(index)
        if indexList:
            if indexList[0] != 0:
                fixedList = np.delete(fixedList, indexList[1:])
                fixedList[indexList[0]] = recoveredRun
            if indexList[0] == 0:
                fixedList = np.delete(fixedList, indexList)
                fixedList = np.append(fixedList, recoveredRun)
        return fixedList

    if experiment == 'Distance':
        if bin == False:
            data90_60, data90_90, data90_120, data10_60, data10_90, data10_120 = ({} for _ in range(6))
            for animal in animal_list:
                data90_60[animal], data90_90[animal], data90_120[animal], data10_60[animal], data10_90[animal], data10_120[animal] = ([] for _ in range(6))
                for session in sorted(matchsession(animal, session_list)):
                    if params[animal, session]['treadmillDist'] == 60:
                        if datatype == 'nb_runs':
                            data90_60[animal] = np.append(data90_60[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_60[animal] = np.append(data10_60[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_60[animal] = np.append(data90_60[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_60[animal] = np.append(data10_60[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                    if params[animal, session]['treadmillDist'] == 90:
                        if datatype == 'nb_runs':
                            data90_90[animal] = np.append(data90_90[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_90[animal] = np.append(data10_90[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_90[animal] = np.append(data90_90[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_90[animal] = np.append(data10_90[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                    if params[animal, session]['treadmillDist'] == 120:
                        if datatype == 'nb_runs':
                            data90_120[animal] = np.append(data90_120[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_120[animal] = np.append(data10_120[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_120[animal] = np.append(data90_120[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_120[animal] = np.append(data10_120[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
            return data90_60, data90_90, data90_120, data10_60, data10_90, data10_120

        if bin == True:
            data60, data90, data120 = ({} for _ in range(3))
            for animal in animal_list:
                data60[animal], data90[animal], data120[animal] = ({bin: [] for bin in range(0, (12))} for i in range(3))
                for session in sorted(matchsession(animal, session_list)):
                    if params[animal, session]['treadmillDist'] == 60:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data60[animal][i] = np.append(data60[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data60[animal][i] = np.append(data60[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                data60[animal][i] = fix_singleRun(data60[animal][i])

                    if params[animal, session]['treadmillDist'] == 90:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data90[animal][i] = np.append(data90[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data90[animal][i] = np.append(data90[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                data90[animal][i] = fix_singleRun(data90[animal][i])

                    if params[animal, session]['treadmillDist'] == 120:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data120[animal][i] = np.append(data120[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data120[animal][i] = np.append(data120[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                data120[animal][i] = fix_singleRun(data120[animal][i])

            return data60, data90, data120

    if experiment == 'TM_ON':
        if bin == False:
            data90_rev20, data90_rev10, data90_rev2, data90_2, data90_10, data90_20, data10_rev20, data10_rev10, data10_rev2, data10_2, data10_10, data10_20 = ({} for _ in range(12))
            for animal in animal_list:
                data90_rev20[animal], data90_rev10[animal], data90_rev2[animal], data90_2[animal], data90_10[animal], data90_20[animal], data10_rev20[animal], data10_rev10[animal], data10_rev2[animal], data10_2[animal], data10_10[animal], data10_20[animal] = ([] for _ in range(12))
                for session in sorted(matchsession(animal, session_list)):
                    if params[animal, session]['treadmillSpeed'] == [-20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20]:
                        if datatype == 'nb_runs':
                            data90_rev20[animal] = np.append(data90_rev20[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_rev20[animal] = np.append(data10_rev20[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_rev20[animal] = np.append(data90_rev20[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_rev20[animal] = np.append(data10_rev20[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                    if params[animal, session]['treadmillSpeed'] == [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]:
                        if datatype == 'nb_runs':
                            data90_rev10[animal] = np.append(data90_rev10[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_rev10[animal] = np.append(data10_rev10[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_rev10[animal] = np.append(data90_rev10[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_rev10[animal] = np.append(data10_rev10[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                    if params[animal, session]['treadmillSpeed'] == [-2,  -2,  -2,  -2,  -2,  -2, - 2,  -2,  -2,  -2,  -2,  -2]:
                        if datatype == 'nb_runs':
                            data90_rev2[animal] = np.append(data90_rev2[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_rev2[animal] = np.append(data10_rev2[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_rev2[animal] = np.append(data90_rev2[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_rev2[animal] = np.append(data10_rev2[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                    if params[animal, session]['treadmillSpeed'] == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]:
                        if datatype == 'nb_runs':
                            data90_2[animal] = np.append(data90_2[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_2[animal] = np.append(data10_2[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_2[animal] = np.append(data90_2[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_2[animal] = np.append(data10_2[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                    if params[animal, session]['treadmillSpeed'] == [10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10]:
                        if datatype == 'nb_runs':
                            data90_10[animal] = np.append(data90_10[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_10[animal] = np.append(data10_10[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_10[animal] = np.append(data90_10[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_10[animal] = np.append(data10_10[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                    if params[animal, session]['treadmillSpeed'] == [20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20]:
                        if datatype == 'nb_runs':
                            data90_20[animal] = np.append(data90_20[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                            data10_20[animal] = np.append(data10_20[animal], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))])
                        else:
                            data90_20[animal] = np.append(data90_20[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
                            data10_20[animal] = np.append(data10_20[animal], [i for i in poolByReward([dataRight[animal, session], dataLeft[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])])
            return data90_rev20, data90_rev10, data90_rev2, data90_2, data90_10, data90_20, data10_rev20, data10_rev10, data10_rev2, data10_2, data10_10, data10_20

        if bin == True:
            datarev20, datarev10, datarev2, data2, data10, data20 = ({} for _ in range(6))
            for animal in animal_list:
                datarev20[animal], datarev10[animal], datarev2[animal], data2[animal], data10[animal], data20[animal] = ({bin: [] for bin in range(0, (12))} for i in range(6))
                for session in sorted(matchsession(animal, session_list)):
                    if params[animal, session]['treadmillSpeed'] == [-20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20]:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                datarev20[animal][i] = np.append(datarev20[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                datarev20[animal][i] = np.append(datarev20[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                datarev20[animal][i] = fix_singleRun(datarev20[animal][i])

                    if params[animal, session]['treadmillSpeed'] == [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                datarev10[animal][i] = np.append(datarev10[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                datarev10[animal][i] = np.append(datarev10[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                datarev10[animal][i] = fix_singleRun(datarev10[animal][i])

                    if params[animal, session]['treadmillSpeed'] == [- 2,  -2,  -2,  -2,  -2,  -2, - 2,  -2,  -2,  -2,  -2,  -2]:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                datarev2[animal][i] = np.append(datarev2[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                datarev2[animal][i] = np.append(datarev2[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                datarev2[animal][i] = fix_singleRun(datarev2[animal][i])

                    if params[animal, session]['treadmillSpeed'] == [2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2]:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data2[animal][i] = np.append(data2[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data2[animal][i] = np.append(data2[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                data2[animal][i] = fix_singleRun(data2[animal][i])

                    if params[animal, session]['treadmillSpeed'] == [10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10]:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data10[animal][i] = np.append(data10[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data10[animal][i] = np.append(data10[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                data10[animal][i] = fix_singleRun(data10[animal][i])
                    if params[animal, session]['treadmillSpeed'] == [20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20]:
                        if datatype == 'nb_runs':
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data20[animal][i] = np.append(data20[animal][i], [b/(int((params[animal, session]['blocks'][i][1]-params[animal, session]['blocks'][i][0])/60)) for b in dataLeft[animal, session].values()][i])
                        else:
                            for i in range(0, len(params[animal, session]['blocks'])):
                                data20[animal][i] = np.append(data20[animal][i], [dataRight[animal, session][i] + dataLeft[animal, session][i]])
                                data20[animal][i] = fix_singleRun(data20[animal][i])
            return datarev20, datarev10, datarev2, data2, data10, data20


def across_session_plot(plot, animal_list, session_list, dataLeft, dataRight, experiment, params, plot_axes, ticks, titles_plot_xaxis_yaxis, datatype, marker, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_title(titles_plot_xaxis_yaxis[0], fontsize=16)
    ax.set_xlabel(titles_plot_xaxis_yaxis[1], fontsize=16)
    ax.set_ylabel(titles_plot_xaxis_yaxis[2], fontsize=16)
    ax.set_xlim(plot_axes[0], plot_axes[1])
    ax.set_ylim(plot_axes[2], plot_axes[3])
    if ticks[0] != []:
        ax.set_xticks(ticks[0])
    if ticks[1] != []:
        ax.set_yticks(ticks[1])
    ax.tick_params(width=1.5, labelsize=12)
    # if experiment == 'TM_ON': ax.tick_params(axis = 'x', rotation = 45)
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.yaxis.set_label_coords(-0.22, 0.5)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(90/250 if plot == "90%" else
                       10/250 if plot == "10%" else
                       0)
    ax.yaxis.label.set_color('dodgerblue' if datatype == 'avgrunspeed' else
                             'red' if datatype == 'runningtime' else
                             'orange' if datatype == 'idletime' else
                             'red'if datatype == 'maxspeed' else 'k')

    a, b, c, d, e, f, g, h, i, j, k, l = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if experiment == 'Distance':
        data90_60, data90_90, data90_120, data10_60, data10_90, data10_120 = separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, False)
        for animal in animal_list:
            if datatype == 'runningtime':
                realdist60, realdist90, realdist120 = ticks[2]
                x = (np.nanmean(realdist60[animal]), np.nanmean(realdist90[animal]), np.nanmean(realdist120[animal]))
                ax.set_xticks([int(np.nanmean([np.nanmean(realdist60[animal]) for animal in animal_list])),
                               int(np.nanmean([np.nanmean(realdist90[animal]) for animal in animal_list])),
                               int(np.nanmean([np.nanmean(realdist120[animal]) for animal in animal_list]))])
                ax.set_xlim(plot_axes[0], plot_axes[1])
            else:
                x = (60, 90, 120)

            if datatype == 'nb_runs':
                a = np.median(data90_60[animal])
                b = np.median(data90_90[animal])
                c = np.median(data90_120[animal])
                d = np.median(data10_60[animal])
                e = np.median(data10_90[animal])
                f = np.median(data10_120[animal])
            else:
                a = np.nanmedian([item for sublist in data90_60[animal] for item in sublist])
                b = np.nanmedian([item for sublist in data90_90[animal] for item in sublist])
                c = np.nanmedian([item for sublist in data90_120[animal] for item in sublist])
                d = np.nanmedian([item for sublist in data10_60[animal] for item in sublist])
                e = np.nanmedian([item for sublist in data10_90[animal] for item in sublist])
                f = np.nanmedian([item for sublist in data10_120[animal] for item in sublist])

            if plot == "90%":
                ax.plot(x, (a, b, c), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
                # ax.errorbar(x, (a, b, c), yerr = (stats.std([item for sublist in data90_60[animal]  for item in sublist]),  stats.std([item for sublist in data90_90[animal]  for item in sublist]), stats.std([item for sublist in data90_120[animal] for item in sublist])), color = marker[animal][0], linestyle=marker[animal][2])
            if plot == "10%":
                ax.plot(x, (d, e, f), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
                # ax.errorbar(x, (d, e, f), yerr = (stats.std([item for sublist in data10_60[animal]  for item in sublist]),  stats.std([item for sublist in data10_90[animal]  for item in sublist]), stats.std([item for sublist in data10_120[animal] for item in sublist])), color = marker[animal][0], linestyle=marker[animal][2])
            if plot == "%":
                ax.plot(x, (d/a, e/b, f/c), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])

    if experiment == 'TM_ON':
        data90_rev20, data90_rev10, data90_rev2, data90_2, data90_10, data90_20, data10_rev20, data10_rev10, data10_rev2, data10_2, data10_10, data10_20 = separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, False)
        for animal in animal_list:
            x = (-20, -10, -2, 2, 10, 20)
            if datatype == 'nb_runs':
                a = np.median(data90_rev20[animal])
                b = np.median(data90_rev10[animal])
                c = np.median(data90_rev2[animal])
                d = np.median(data90_2[animal])
                e = np.median(data90_10[animal])
                f = np.median(data90_20[animal])

                g = np.median(data10_rev20[animal])
                h = np.median(data10_rev10[animal])
                i = np.median(data10_rev2[animal])
                j = np.median(data10_2[animal])
                k = np.median(data10_10[animal])
                l = np.median(data10_20[animal])
            else:
                a = np.nanmedian([item for sublist in data90_rev20[animal] for item in sublist])
                b = np.nanmedian([item for sublist in data90_rev10[animal] for item in sublist])
                c = np.nanmedian([item for sublist in data90_rev2[animal] for item in sublist])
                d = np.nanmedian([item for sublist in data90_2[animal] for item in sublist])
                e = np.nanmedian([item for sublist in data90_10[animal] for item in sublist])
                f = np.nanmedian([item for sublist in data90_20[animal] for item in sublist])

                g = np.nanmedian([item for sublist in data10_rev20[animal] for item in sublist])
                h = np.nanmedian([item for sublist in data10_rev10[animal] for item in sublist])
                i = np.nanmedian([item for sublist in data10_rev2[animal] for item in sublist])
                j = np.nanmedian([item for sublist in data10_2[animal] for item in sublist])
                k = np.nanmedian([item for sublist in data10_10[animal] for item in sublist])
                l = np.nanmedian([item for sublist in data10_20[animal] for item in sublist])

            if plot == "90%":
                ax.plot(x, (a, b, c, d, e, f), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "10%":
                ax.plot(x, (g, h, i, j, k, l), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "%":
                ax.plot(x, (g/a, h/b, i/c, j/d, k/e, l/f), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
    return ax


def stars(p, maxasterix=3):
    if type(p) is str:
        text = p
    
    else:
        if p < .001:
            text = f"$p$<0.001"
        elif p < .01:
            text = f"$p$={p:.1e}"
        elif p <= .06:
            text = f"$p$={p:.3f}"
        else:
            text = f"$p$={p:.2f}"
        # text = ''
        # sig = .05

        # while p < sig:
        #     text += '*'
        #     sig /= 10.

        #     if maxasterix and len(text) == maxasterix:
        #         break

        # if len(text) == 0:
        #     text = f'n.s. $p$ = {p:.3f}'

    return text


def barplot_annotate_brackets(ax, num1, num2, data, center, height, dh=.05, barh=.05, fs=5, maxasterix=3):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    text = stars(data, maxasterix)

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    if ly > 0 and ry > 0:
        y = max(ly, ry) + dh

    else:
        y = min(ly, ry) + dh

    mid = ((lx+rx)/2, y+barh*1.25)
    if dh < 0:
        kwargs = dict(ha='center', va='top')
    else:
        kwargs = dict(ha='center', va='bottom')

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]


    ax.plot(barx, bary, c='black', lw=0.5)

    if fs is not None:
        if '*' in text:
            kwargs['fontsize'] = 7
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)


def plot_kde(data, bandwidth=1, ax=None, color='k', xx=[-4, 4]):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    data = np.array(data).reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(data)

    # Compute log density scores
    xx = np.linspace(xx[0], xx[1], 1000).reshape(-1, 1)
    log_densities = kde.score_samples(xx)
    densities = np.exp(log_densities)
    
    ax.plot(xx, densities, color=color, lw=1, alpha=0.8)
    ax.fill_between(xx.flatten(), densities, color=color, alpha=0.1, lw=0)