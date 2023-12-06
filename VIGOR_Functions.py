import copy
try:
    import dabest
except:
    print("dabest not installed")
import datetime
import fnmatch
import glob
from IPython.display import Image, display, clear_output
import itertools
from itertools import groupby
import matplotlib.cbook
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import os
import pandas as pd
import pickle
import platform
import re
import scipy
from scipy import stats
from scipy.ndimage import gaussian_filter as smooth
from scipy.signal import find_peaks
import sys
import time
import warnings
import matplotlib.ticker as mticker
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
startTimeNotebook = datetime.datetime.now()


"""
###----------------------------------------------------------------------------
### Utility/low_level computation functions()
###----------------------------------------------------------------------------
"""


# conversion
def inch2cm(value): return value * 2.54
def cm2inch(value): return value / 2.54


# new px to cm conversion. To correct camera lens distorsion (pixels in the
# center of the treadmill are more precise than the ones located at the
# extremities), filter applied in LabView, and conversion should be uniform
# now, 11 px is equal to 1 cm at every point of the treadmill.
def datapx2cm(list):
    array = []
    for pos in list:
        if pos == 0:
            array.append(pos)
        elif pos > 0 and pos < 1300:
            array.append(pos/11)
        else:
            array.append(pos)
            print("might have error in position", pos)
    return array


# function to split lists --> used to split the raw X position array into
# smaller arrays (runs and stays). Later in the code we modify the
# array and change some values to 0, which will be used as cutting points.
def split_a_list_at_zeros(List):
    return [list(g) for k, g in groupby(List, key=lambda x:x != 0) if k]


# function to open and read from the .position files using pandas, specify
# the path of the file to open, the column that you want
# to extract from, and the extension of the file
def read_csv_pandas(path, Col=None, header=None):
    #  verify that the file exists
    if not os.path.exists(path):
        print("No file %s" % path)
        return []
    try:  # open the file
        csvData = pd.read_csv(path, header=header,
                              delim_whitespace=True, low_memory=False)
    except ValueError:
        print("%s not valid (usually empty)" % path)
        return []
        # verify that the column that we specified is not empty,
        # and return the values
    if Col is not None:
        return csvData.values[:, Col[0]]
    else:
        return csvData


# cuts session in bins
def bin_session(animal, session, data_to_cut, data_template, bins):
    output = {}
    bincount = 0
    for timebin in bins:
        if timebin[0] == 0:
            start_of_bin = 0
        else:
            start_of_bin = int(np.where(data_template[animal, session] == timebin[0])[0])+1
        end_of_bin = int(np.where(data_template[animal, session] == timebin[1])[0])+1
        output[bincount] = data_to_cut[animal, session][start_of_bin:end_of_bin]
        bincount += 1
    return output


# function to read the parameters for each rat for the session in the
# behav.param file. Specify the name of the parameter that you want
# to get from the file and optionally the value type that you want.
# File path is not an option, maybe change that. Dataindex is in case
# you don't only want the last value in line, so you can choose which
# value you want using its index --maybe add the
# option to choose a range of values.
def read_params(root, animal, session, paramName, dataindex=-1, valueType=str):
    # define path of the file
    behav = root + os.sep+animal + os.sep+"Experiments" + \
            os.sep + session + os.sep + session + ".behav_param"
    # check if it exists
    if not os.path.exists(behav):
        print("No file %s" % behav)
    # check if it is not empty
    # if os.stat(behav).st_size == 0:
        # print("File empty %s" % behav)
    with open(behav, "r") as f:
        # scan the file for a specific parameter, if the name of
        # the parameter is there, get the value
        for line in f:
            if valueType is str:
                if paramName in line:
                    # get the last value of the line [-1], values are
                    # separated with _blanks_ with the .split() function
                    return int(line.split()[dataindex])
            if valueType is float:
                if paramName in line:
                    return float(line.split()[dataindex])
            else:
                if paramName in line:
                    return str(line.split()[dataindex])


# 2022_05_04 LV: added brain status of the animal (normal, lesion, cno, saline) in behav.params. 
# This is a fix to consign it in antecedent sessions. I think it fixed them all.
def FIXwrite_params(root, animal, session):
    # animal = "RatF02"
    # for session in sorted(matchsession(animal, lesionrev20+lesionrev10+lesionrev2+lesion2+lesion10+lesion20)): #lesiontrain+lesion60+lesion90+lesion120
    #     FIXwrite_params(root, animal, session)
    behav = root + os.sep+animal + os.sep+"Experiments" + os.sep + session + os.sep + session + ".behav_param"
    if not os.path.exists(behav):
        print("No file %s" % behav)
    alreadywritten=False
    with open(behav, "r") as f:
        for line in f:
            if "brainstatus" in line:
                alreadywritten = True
    if not alreadywritten:
        with open(behav, "a") as f: f.write("\nbrainstatus normal")


# save data as pickle
def save_as_pickle(root, data, animal, session, name):
    sessionPath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session
    folderPath = os.path.join(sessionPath, "Analysis")
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    filePath = os.path.join(folderPath, name)
    pickle.dump(data, open(filePath, "wb"))


# load data that has been pickled
def get_from_pickle(root, animal, session, name):
    sessionPath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session
    analysisPath = os.path.join(sessionPath, "Analysis")
    picklePath = os.path.join(analysisPath, name)
    # if not re.do, and there is a pickle, try to read it
    if os.path.exists(picklePath):
        try:
            data = pickle.load(open(picklePath, "rb"))
            return data
        except:
            print("error")
            pass


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


# save plot as png
def save_plot_as_png(filename, dpi='figure',
                     transparent=True, background='auto'):
    folderPath = os.path.join(os.getcwd(), "Figures")
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    filePath = os.path.join(folderPath, filename)
    plt.savefig(filePath, dpi=dpi, transparent=transparent,
                facecolor=background, edgecolor=background)


# only display one legend when there is duplicates
# (e.g. we don't want one label per run)
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


# function to compute moving average, used to see the eventual acquisition bugs
def movinavg(interval, window_size):
    if window_size != 0:
        window = np.ones(int(window_size))/float(window_size)
    else:
        print("Error: Window size == 0")
    return np.convolve(interval, window, 'same')


# same with median, used to compute moving threshold for lick detection
def movinmedian(interval, window_size):
    if window_size != 0:
        window = int(window_size)
    else:
        print("Error: Window size == 0")
    val = pd.Series(interval)
    return val.rolling(window).median()


def reversemovinmedian(interval, window_size):
    if window_size != 0:
        window = int(window_size)
    else:
        print("Error: Window size == 0")
    val = pd.Series(interval[::-1])
    return list(reversed(val.rolling(window).median()))


# function to compute speed array based on position and time arrays
def compute_speed(dataPos, dataTime):  # speed only computed along X axis. Compute along X AND Y axis?
    rawdata_speed = {}
    deltaXPos = (np.diff(dataPos))
    deltaTime = (np.diff(dataTime))
    rawdata_speed = np.divide(deltaXPos, deltaTime)
    rawdata_speed = np.append(rawdata_speed, 0)
    return rawdata_speed.astype('float32')
    # working on ragged arrays so type of the array may
    # have to be modified from object to float32


def dirty_acceleration(array):
    return [np.diff(a)/0.04 for a in array]


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


# util func to print in a given color
# print text with custom colors (text+background)
# colorprint("texxxxt", (R, G, B), (R, G, B)), with R, G, B [0-1]
def colorprint(text, color, backgroundcolor=None):
    if isinstance(text, str) and isinstance(color, tuple):
        if any(v > 1 or v < 0 for v in color):
            print("RGB must be [0-1] not [0-255]")
        else:
            if backgroundcolor == None:
                return "\033[38;2;{0};{1};{2}m{3}".format(int(color[0]*255),
                                                          int(color[1]*255),
                                                          int(color[2]*255),
                                                          text)
            else:
                return "\033[38;2;{0};{1};{2}m\033[48;2;{3};{4};{5}m{6}".format(
                    int(color[0]*255), int(color[1]*255), int(color[2]*255),
                    int(backgroundcolor[0]*255),
                    int(backgroundcolor[1]*255),
                    int(backgroundcolor[2]*255), text)
    else:
        print("error: check input type")


# util funct to print a progress bar
def update_progress(progress, root):
    barLength = 50
    animalList = [os.path.basename(path) for path in sorted(glob.glob(root+"/Rat*"))]

    def status_update(progress):
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "Error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Something's WRONG...\r\n"
        if progress < 1 and progress >= 0:
            status = "Computing..."
        if progress >= 0.999:
            progress = 1
            status = 'Done ✓ \r'
        return status

    # text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "~~~(___C'>" + "-"*(barLength-block), int(round(progress*100)), progressstatus(progress, session))
    # text = ("\r"   + animalList[0] + "{0}".format(" " * int(round(barLength * progress[0])) + "          __QQ" + " " * (barLength - int(round(barLength * progress[0])))) + " " * 180 + "Progress: [{0}] {1}% {2}".format(" " * int(round(barLength * progress[0])) + " ~~~(_)_\">" + " " * (barLength-int(round(barLength * progress[0]))), int(round(progress[0] * 100)), status_update(progress[0])) +
    #         "\n\n" + animalList[1] + "{0}".format(" " * int(round(barLength * progress[1])) + "          __QQ" + " " * (barLength - int(round(barLength * progress[1])))) + " " * 180 + "Progress: [{0}] {1}% {2}".format(" " * int(round(barLength * progress[1])) + " ~~~(_)_\">" + " " * (barLength-int(round(barLength * progress[1]))), int(round(progress[1] * 100)), status_update(progress[1])) +
    #         "\n\n" + animalList[2] + "{0}".format(" " * int(round(barLength * progress[2])) + "          __QQ" + " " * (barLength - int(round(barLength * progress[2])))) + " " * 180 + "Progress: [{0}] {1}% {2}".format(" " * int(round(barLength * progress[2])) + " ~~~(_)_\">" + " " * (barLength-int(round(barLength * progress[2]))), int(round(progress[2] * 100)), status_update(progress[2])) +
    #         "\n\n" + animalList[3] + "{0}".format(" " * int(round(barLength * progress[3])) + "          __QQ" + " " * (barLength - int(round(barLength * progress[3])))) + " " * 180 + "Progress: [{0}] {1}% {2}".format(" " * int(round(barLength * progress[3])) + " ~~~(_)_\">" + " " * (barLength-int(round(barLength * progress[3]))), int(round(progress[3] * 100)), status_update(progress[3])) +
    #         "\n\n" + animalList[4] + "{0}".format(" " * int(round(barLength * progress[4])) + "          __QQ" + " " * (barLength - int(round(barLength * progress[4])))) + " " * 180 + "Progress: [{0}] {1}% {2}".format(" " * int(round(barLength * progress[4])) + " ~~~(_)_\">" + " " * (barLength-int(round(barLength * progress[4]))), int(round(progress[4] * 100)), status_update(progress[4])) +
    #         "\n\n" + animalList[5] + "{0}".format(" " * int(round(barLength * progress[5])) + "          __QQ" + " " * (barLength - int(round(barLength * progress[5])))) + " " * 180 + "Progress: [{0}] {1}% {2}".format(" " * int(round(barLength * progress[5])) + " ~~~(_)_\">" + " " * (barLength-int(round(barLength * progress[5]))), int(round(progress[5] * 100)), status_update(progress[5])) + "\n\n")

    text = ("\r")
    for i, animal in enumerate(animalList):
        text += animalList[i] + "{0}".format(" " * int(round(barLength * progress[i])) +
        "          __QQ" + " " * (barLength - int(round(barLength * progress[i])))) + " " * 180 + \
        "Progress: [{0}] {1}% {2}".format(" " * int(round(barLength * progress[i])) + " ~~~(_)_\">" +
        " " * (barLength-int(round(barLength * progress[i]))),
        int(round(progress[i] * 100)), status_update(progress[i])) + "\n\n"

    sys.stdout.write(text)
    sys.stdout.flush()


# select specified session among all the session in datapath
# def matchsession(animal, sessionlist):
#     list = [session for session in sessionlist if animal == session[0:6]]
#     return (list)
def matchsession(animal, sessionlist, AMPM=False):
    list = [session for session in sessionlist if animal == session[0:6]]
    if AMPM:
        list = [session for session in list if int(session[-8:-6]) < 14] if AMPM == "AM" else [session for session in list if int(session[-8:-6]) > 14]
    return (list)


# select a random session for specified animal
def randomsession(animal, sessionlist):
    list = [session for session in sessionlist if animal == session[0:6]]
    return(list)


# in action sequence, get block number based on beginning of action 
def get_block(t_0):
    if 0 <= t_0 <= 300:
        block = 0
    elif 300 < t_0 <= 600:
        block = 1
    elif 600 < t_0 <= 900:
        block = 2
    elif 900 < t_0 <= 1200:
        block = 3
    elif 1200 < t_0 <= 1500:
        block = 4
    elif 1500 < t_0 <= 1800:
        block = 5
    elif 1800 < t_0 <= 2100:
        block = 6
    elif 2100 < t_0 <= 2400:
        block = 7
    elif 2400 < t_0 <= 2700:
        block = 8
    elif 2700 < t_0 <= 3000:
        block = 9
    elif 3000 < t_0 <= 3300:
        block = 10
    elif 3300 < t_0 <= 3600:
        block = 11
    elif 0 > t_0:
        block = None
    return block


# in action sequence, cut full action sequence into corresponding blocks
def recut(data_to_cut, data_template):
    output = []
    start_of_bin = 0
    for i, _ in enumerate(data_template):
        end_of_bin = start_of_bin + len(data_template[i])
        output.append(data_to_cut[start_of_bin: end_of_bin])
        start_of_bin = end_of_bin
    return output


"""
###----------------------------------------------------------------------------
### INDIVIDUAL FIGURES
###----------------------------------------------------------------------------
"""


# This function plots the base trajectory of the rat. Parameters are time:
# time data, position : X position data, lickL/R, lick data,
# maxminstep for x and y axis, color and marker of the plot,
# width of the axis, and x y labels
def plot_BASEtrajectory(time, position, lickLeft, lickRight, maxminstep,
                        maxminstep2, color=[], marker=[],
                        linewidth=[], xyLabels=["N", "Bins"]):

    plt.plot(time, position, color=color[0],
             marker=marker[0], linewidth=linewidth[0])
    # lick data, plot position in which the animal licked else None
    plt.plot(time, [None if x == 0 else x for x in lickLeft], color=color[1],
             marker=marker[1], markersize=marker[2])
    plt.plot(time, [None if x == 0 else x for x in lickRight], color=color[1],
             marker=marker[1], markersize=marker[2])

    # plot parameters
    traj = plt.gca()
    traj.set_xlim(maxminstep[0] - maxminstep[2],
                  maxminstep[1] + maxminstep[2])
    traj.set_ylim(maxminstep2[0] - maxminstep2[2],
                  maxminstep2[1] + maxminstep2[2])
    traj.set_xlabel(xyLabels[1], fontsize=12, labelpad=0)
    traj.set_ylabel(xyLabels[0], fontsize=12, labelpad=-1)
    traj.xaxis.set_ticks_position('bottom')
    traj.yaxis.set_ticks_position('left')
    traj.get_xaxis().set_tick_params(direction='out', pad=2)
    traj.get_yaxis().set_tick_params(direction='out', pad=2)
    traj.spines['top'].set_color("none")
    traj.spines['right'].set_color("none")
    return traj


def plot_BASEtrajectoryV2(animal, session, time, running_Xs, idle_Xs, lickL, lickR,
                          rewardProbaBlock, blocks, barplotaxes,
                          xyLabels=[" ", " ", " ", " "],
                          title=[None], linewidth=1):
    ax1 = plt.gca()
    for i in range(0, len(blocks)):
        plt.axvspan(blocks[i][0], blocks[i][1],
                    color='grey', alpha=rewardProbaBlock[i]/250,
                    label="%reward: " + str(rewardProbaBlock[i])
                    if (i == 0 or i == 1) else "")
    plt.plot(time, running_Xs, label="run", color="dodgerblue", linewidth=1)
    plt.plot(time, idle_Xs, label="wait", color="orange", linewidth=1)
    plt.plot(time, [None if x == 0 else x for x in lickL],
             color="b", marker="o", markersize=1)
    plt.plot(time, [None if x == 0 else x for x in lickR],
             color="b", marker="o", markersize=1)
    ax1.set_xlabel(xyLabels[0], fontsize=xyLabels[6])
    ax1.set_ylabel(xyLabels[2], fontsize=xyLabels[6])
    ax1.set_xlim([barplotaxes[0], barplotaxes[1]])
    ax1.set_ylim([barplotaxes[0], barplotaxes[3]+30])
    ax1.spines['bottom'].set_linewidth(linewidth[0])
    ax1.spines['left'].set_linewidth(linewidth[0])
    ax1.spines['top'].set_color("none")
    ax1.spines['right'].set_color("none")
    ax1.tick_params(width=2, labelsize=xyLabels[7])
    # x_ticks = np.arange(1800, 3300, 300)
    # ax1.set_xticks(x_ticks)
    # ax1.set_xticklabels([int(val / 60) for val in ax1.get_xticks().tolist()])
    return ax1


# plot old boundaries run stay
def plot_peak(data, animal, session, leftBoundaryPeak, rightBoundaryPeak, kde,
              maxminstep, maxminstep2, marker=[], xyLabels=["N", "Bins"]):
    # fig, ax = plt.subplots(figsize=(3,6))
    bins = np.arange(120)
    xx = np.linspace(0, 120, 120)
    xline1 = [leftBoundaryPeak, leftBoundaryPeak]
    xline2 = [rightBoundaryPeak, rightBoundaryPeak]
    border = 5
    yline = [0, 0.01]
    peak = plt.gca()
    if platform.system() == 'Darwin':
        peak.hist(data, normed=True, bins=bins, alpha=0.3,
                  orientation='horizontal')  # bugged on linux, working on mac

    # plot kde + boundaries
    peak.plot(kde(xx), xx, color='r')
    peak.plot(yline, xline1, ":", color='k')
    peak.plot(yline, xline2, ":", color='k')
    peak.plot(yline, [xline1[0] + border, xline1[0] + border],
              ":", c='k', alpha=0.5)
    peak.plot(yline, [xline2[0] - border, xline2[0] - border],
              ":", c='k', alpha=0.5)
    # configure plot
    peak.set_xlim(maxminstep[0] - maxminstep[2],
                  maxminstep[1] + maxminstep[2])
    peak.set_ylim(maxminstep2[0] - maxminstep2[2],
                  maxminstep2[1] + maxminstep2[2])
    peak.set_xlabel(xyLabels[1], fontsize=12, labelpad=0)
    peak.set_ylabel(xyLabels[0], fontsize=12, labelpad=-1)
    peak.spines['top'].set_color("none")
    peak.spines['left'].set_color("none")
    peak.spines['right'].set_color("none")
    peak.yaxis.set_ticks_position('left')
    peak.xaxis.set_ticks_position('bottom')
    peak.get_xaxis().set_tick_params(direction='out', pad=2)
    peak.get_yaxis().set_tick_params(direction='out', pad=2)
    peak.axes.get_yaxis().set_visible(False)
    return peak


# function to plot the tracks of each runs and stays
def plot_tracks(animal, session, posdataRight, timedataRight, bounds, xylim,
                color, xyLabels=[" ", " ", " ", " "], title=[None], linewidth=1):
    tracksplot = plt.gca()
    tracksplot.set_title(title[0], fontsize=title[1], pad=50)
    for i, j in zip(posdataRight, timedataRight):
        plt.plot(np.subtract(j, j[0]), i, color=color[0], linewidth=0.3,
                 label="Good Item" if i == posdataRight[0] else "")
    tracksplot.set_title(title[0], fontsize=title[1])
    tracksplot.set_xlabel(xyLabels[0], fontsize=xyLabels[2])
    tracksplot.set_ylabel(xyLabels[1], fontsize=xyLabels[2])
    tracksplot.set_xlim([xylim[0], xylim[1]])
    tracksplot.set_ylim([xylim[2], xylim[3]])
    tracksplot.spines['top'].set_color("none")
    tracksplot.spines['right'].set_color("none")
    xline1 = [bounds[0], bounds[0]]
    xline2 = [bounds[1], bounds[1]]
    yline = [0, 20]
    plt.plot(yline, xline1, ":", color='k')
    plt.plot(yline, xline2, ":", color='k')
    tracksplot.legend()
    return tracksplot


# function to plot the cumulative distribution of the run speeds and stay times
def cumul_plot(dataRight, dataLeft, barplotaxes, maxminstepbin,
               scatterplotaxes, legend, color, xyLabels=["", "", "", ""],
               title=[None], linewidth=1):
    cumul = plt.gca()
    custom_legend = [Line2D([0], [0], color=color[0], lw=legend[3]),
                     Line2D([0], [0], color=color[1], lw=legend[2])]
    plt.hist(dataRight,
             np.arange(maxminstepbin[0], maxminstepbin[1], maxminstepbin[2]),
             weights=np.ones_like(dataRight)/float(len(dataRight)), color=color[0],
             histtype='step', cumulative=True, linewidth=legend[3])
    plt.hist(dataLeft,
             np.arange(maxminstepbin[0], maxminstepbin[1], maxminstepbin[2]),
             weights=np.ones_like(dataLeft)/float(len(dataLeft)), color=color[1],
             histtype='step', cumulative=True, linewidth=legend[2])
    cumul.set_title(title[0], fontsize=title[1], pad=50)
    cumul.set_xlabel(xyLabels[0], fontsize=xyLabels[2])
    cumul.set_ylabel(xyLabels[1], fontsize=xyLabels[2])
    cumul.set_xlim([barplotaxes[0], barplotaxes[1]])
    cumul.set_ylim([barplotaxes[2], barplotaxes[3]])
    cumul.spines['bottom'].set_linewidth(linewidth[0])
    cumul.spines['left'].set_linewidth(linewidth[0])
    cumul.spines['top'].set_color("none")
    cumul.spines['right'].set_color("none")
    cumul.tick_params(width=2, labelsize=xyLabels[2])
    # plt.tight_layout(pad=0.5)
    cumul.legend(custom_legend, [legend[0], legend[1]],
                 bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                 mode="expand", borderaxespad=0., frameon=False)
    return cumul


# function to plot the scatter data of run times and stay times
def distribution_plot(dataRight, dataLeft, barplotaxes, maxminstepbin,
                      scatterplotaxes, legend, color, xyLabels=["", "", "", ""],
                      title=[None], linewidth=1):
    distr = plt.gca()
    plt.scatter(np.random.normal(1, 0.05, len(dataRight)), dataRight,
                s=20, color=color[0], marker="$\u25ba$", label=legend[0])
    plt.scatter(np.random.normal(2, 0.05, len(dataLeft)), dataLeft,
                s=20, color=color[1], marker="$\u25c4$", label=legend[1])

    plt.scatter(1.2, np.mean(dataRight), s=25, color=color[0])
    plt.scatter(2.2, np.mean(dataLeft), s=25, color=color[1])
    plt.boxplot(dataRight, positions=[1.35])
    plt.boxplot(dataLeft, positions=[2.35])
    distr.set_xlabel(xyLabels[1], fontsize=xyLabels[4])
    distr.set_ylabel(xyLabels[0], fontsize=xyLabels[4])
    distr.set_title(title[0], fontsize=title[1], pad=50)
    distr.set_xlim([scatterplotaxes[0], scatterplotaxes[1]])
    distr.set_ylim([scatterplotaxes[2], scatterplotaxes[3]])
    distr.set_xticks([1, 2])
    distr.set_xticklabels([xyLabels[2], xyLabels[3]], fontsize=xyLabels[5])
    distr.spines['bottom'].set_linewidth(linewidth[0])
    distr.spines['left'].set_linewidth(linewidth[0])
    distr.spines['top'].set_color("none")
    distr.spines['right'].set_color("none")
    distr.tick_params(width=2, labelsize=xyLabels[5])
    handles, labels = distr.get_legend_handles_labels()
    distr.legend()
    # distr.legend([handles[0], handles[1],],
    # [legend[0], legend[2], legend[1], legend[3]], bbox_to_anchor = (0., 1.02, 1., .102),
    # loc='lower left', ncol=2, mode="expand", borderaxespad=0., frameon = False)
    return distr


# plot rat speed along run
def plot_speed(animal, session, posdataRight, timedataRight, bounds, xylim,
               xyLabels=[" ", " ", " ", " "], title=[None], linewidth=1):
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
    ax.set_title(title[0], fontsize=title[1])
    ax.set_xlabel(xyLabels[0], fontsize=xyLabels[2])
    ax.set_ylabel(xyLabels[1], fontsize=xyLabels[2])
    ax.set_xlim([xylim[0], xylim[1]])
    ax.set_ylim([xylim[2], xylim[3]])
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    xline1 = [bounds[0], bounds[0]]
    xline2 = [bounds[1], bounds[1]]
    yline = [0, 20]
    plt.plot(yline, xline1, ":", color='k')
    plt.plot(yline, xline2, ":", color='k')
    ax.legend()
    return ax


# plot rat acceleration along run
def plot_acc(animal, session, posdataRight, timedataRight, bounds,
             xylim, xyLabels=[" ", " ", " ", " "], title=[None], linewidth=1):
    ax = plt.gca()
    for i, j in zip(posdataRight, timedataRight):
        plt.plot(np.subtract(j, j[0]), i, color='g', linewidth=0.3)
    ax.set_title(title[0], fontsize=title[1])
    ax.set_xlabel(xyLabels[0], fontsize=xyLabels[2])
    ax.set_ylabel(xyLabels[1], fontsize=xyLabels[2])
    ax.set_xlim([xylim[0], xylim[1]])
    ax.set_ylim([xylim[2], xylim[3]])
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    xline1 = [bounds[0], bounds[0]]
    xline2 = [bounds[1], bounds[1]]
    yline = [0, 20]
    plt.plot(yline, xline1, ":", color='k')
    plt.plot(yline, xline2, ":", color='k')
    ax.legend()
    return ax


# plot per block
def plot_figBin(data, rewardProbaBlock, blocks, barplotaxes, color, stat,
                xyLabels=[" ", " ", " ", " "], title=[None], linewidth=1, scatter=False, binplot=False):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    if not binplot:
        binplot = plt.gca()
    for i in range(0, len(blocks)):
        binplot.axvspan(blocks[i][0]/60, blocks[i][1]/60, color='grey', alpha=rewardProbaBlock[i]/250, label="%reward: " + str(rewardProbaBlock[i]) if (i == 0 or i == 1) else "")
        if scatter:
            binplot.scatter(np.random.normal(((blocks[i][1] + blocks[i][0])/120), 1, len(data[i])), data[i], s=5, color=color[0])

    if stat == "Avg. ":
        binplot.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.mean(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])
        if isinstance(data[0], list):
            binplot.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.mean(data[i]) for i in range(0, len(blocks))], yerr=[stats.sem(data[i]) for i in range(0, len(blocks))], fmt='o', color=color[0], ecolor='black', elinewidth=1, capsize=0);

    elif stat == "Med. ":
        binplot.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.median(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])
        if isinstance(data[0], list):
            binplot.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.median(data[i]) for i in range(0, len(blocks))], yerr=[stats.sem(data[i]) for i in range(0, len(blocks))], fmt='o', color=color[0], ecolor='black', elinewidth=1, capsize=3);

    binplot.set_title(title[0], fontsize=title[1])
    binplot.set_xlabel(xyLabels[0], fontsize=xyLabels[2])
    binplot.set_ylabel(stat + xyLabels[1], fontsize=xyLabels[2])
    binplot.set_xlim([barplotaxes[0], barplotaxes[1]])
    binplot.set_ylim([barplotaxes[2], barplotaxes[3]])
    binplot.spines['bottom'].set_linewidth(linewidth[0])
    binplot.spines['left'].set_linewidth(linewidth[0])
    binplot.spines['top'].set_color("none")
    binplot.spines['right'].set_color("none")
    binplot.tick_params(width=2, labelsize=xyLabels[3])
    return binplot

    # plot per block
def plot_figBinNew(data, rewardProbaBlock, blocks, barplotaxes, color, stat,
                xyLabels=[" ", " ", " ", " "], title="", scatter=False, ax=False):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    if not ax:
        ax = plt.gca()
    for i in range(0, len(blocks)):
        ax.axvspan(blocks[i][0]/60, blocks[i][1]/60, color='grey', alpha=rewardProbaBlock[i]/250, label="%reward: " + str(rewardProbaBlock[i]) if (i == 0 or i == 1) else "")
        if scatter:
            ax.scatter(np.random.normal(((blocks[i][1] + blocks[i][0])/120), 1, len(data[i])), data[i], s=5, color=color[0])

    if stat == "Avg. ":
        ax.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.mean(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])
        if isinstance(data[0], list):
            ax.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.mean(data[i]) for i in range(0, len(blocks))], yerr=[stats.sem(data[i]) for i in range(0, len(blocks))], fmt='o', color=color[0], ecolor='black', elinewidth=1, capsize=0);

    elif stat == "Med. ":
        ax.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.median(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])
        if isinstance(data[0], list):
            ax.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.median(data[i]) for i in range(0, len(blocks))], yerr=[stats.sem(data[i]) for i in range(0, len(blocks))], fmt='o', color=color[0], ecolor='black', elinewidth=1, capsize=3);

    ax.set_title(title)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(stat + xyLabels[1])
    ax.set_xlim([barplotaxes[0], barplotaxes[1]])
    ax.set_ylim([barplotaxes[2], barplotaxes[3]])
    return ax

# plot per block
def plot_figBinVaria(data, rewardProbaBlock, blocks, barplotaxes, color,
                xyLabels=[" ", " ", " ", " "], title=[None], linewidth=1, scatter=False, binplot=False, stat="std"):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    if not binplot:
        binplot = plt.gca()
    for i in range(0, len(blocks)):
        binplot.axvspan(blocks[i][0]/60, blocks[i][1]/60, color='grey', alpha=rewardProbaBlock[i]/250, label="%reward: " + str(rewardProbaBlock[i]) if (i == 0 or i == 1) else "")
        if scatter:
            binplot.scatter(np.random.normal(((blocks[i][1] + blocks[i][0])/120), 1, len(data[i])), data[i], s=5, color=color[0])

    # binplot.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [stats.sem(data[i]) for i in range(0, len(blocks))], marker='o', ms=5, linewidth=1, color=color[0])
    if stat == 'std':
        binplot.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.std(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])
    if stat == 'sem':      
        binplot.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [stats.sem(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])



    binplot.set_title(title[0], fontsize=title[1])
    binplot.set_xlabel(xyLabels[0], fontsize=xyLabels[2])
    binplot.set_ylabel(stat + xyLabels[1], fontsize=xyLabels[2])
    binplot.set_xlim([barplotaxes[0], barplotaxes[1]])
    binplot.set_ylim([barplotaxes[2], barplotaxes[3]])
    binplot.spines['bottom'].set_linewidth(linewidth[0])
    binplot.spines['left'].set_linewidth(linewidth[0])
    binplot.spines['top'].set_color("none")
    binplot.spines['right'].set_color("none")
    binplot.tick_params(width=2, labelsize=xyLabels[3])
    return binplot

# plot block per %reward
def plot_figBinMean(ax, dataLeft, dataRight, color, ylim):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    two_groups_unpaired = dabest.load(pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in {
            "10%\nreward": dataLeft, "90%\nreward": dataRight}.items()])),
            idx=("10%\nreward", "90%\nreward"), resamples=5000)
    two_groups_unpaired.mean_diff.plot(
        ax=ax, raw_marker_size=7, es_marker_size=5,
        custom_palette={"10%\nreward": color[0], "90%\nreward": color[0]},
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


# different linestyle depending on brain condition of the animal
def brainstatus_plot(status):
    if status == "lesion":
        return 'dotted'
    elif status == "CNO":
        return 'dotted'
    elif status == "saline":
        return 'solid'
    elif status == "normal":
        return 'solid'
    else:
        return 'solid'


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
                a = np.mean(data90_60[animal])
                b = np.mean(data90_90[animal])
                c = np.mean(data90_120[animal])
                d = np.mean(data10_60[animal])
                e = np.mean(data10_90[animal])
                f = np.mean(data10_120[animal])
            else:
                a = np.nanmean([item for sublist in data90_60[animal] for item in sublist])
                b = np.nanmean([item for sublist in data90_90[animal] for item in sublist])
                c = np.nanmean([item for sublist in data90_120[animal] for item in sublist])
                d = np.nanmean([item for sublist in data10_60[animal] for item in sublist])
                e = np.nanmean([item for sublist in data10_90[animal] for item in sublist])
                f = np.nanmean([item for sublist in data10_120[animal] for item in sublist])

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
                a = np.mean(data90_rev20[animal])
                b = np.mean(data90_rev10[animal])
                c = np.mean(data90_rev2[animal])
                d = np.mean(data90_2[animal])
                e = np.mean(data90_10[animal])
                f = np.mean(data90_20[animal])

                g = np.mean(data10_rev20[animal])
                h = np.mean(data10_rev10[animal])
                i = np.mean(data10_rev2[animal])
                j = np.mean(data10_2[animal])
                k = np.mean(data10_10[animal])
                l = np.mean(data10_20[animal])
            else:
                a = np.nanmean([item for sublist in data90_rev20[animal] for item in sublist])
                b = np.nanmean([item for sublist in data90_rev10[animal] for item in sublist])
                c = np.nanmean([item for sublist in data90_rev2[animal] for item in sublist])
                d = np.nanmean([item for sublist in data90_2[animal] for item in sublist])
                e = np.nanmean([item for sublist in data90_10[animal] for item in sublist])
                f = np.nanmean([item for sublist in data90_20[animal] for item in sublist])

                g = np.nanmean([item for sublist in data10_rev20[animal] for item in sublist])
                h = np.nanmean([item for sublist in data10_rev10[animal] for item in sublist])
                i = np.nanmean([item for sublist in data10_rev2[animal] for item in sublist])
                j = np.nanmean([item for sublist in data10_2[animal] for item in sublist])
                k = np.nanmean([item for sublist in data10_10[animal] for item in sublist])
                l = np.nanmean([item for sublist in data10_20[animal] for item in sublist])

            if plot == "90%":
                ax.plot(x, (a, b, c, d, e, f), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "10%":
                ax.plot(x, (g, h, i, j, k, l), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "%":
                ax.plot(x, (g/a, h/b, i/c, j/d, k/e, l/f), marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
    return ax


def across_Binsession_plot(plot, animal_list, session_list, dataLeft, dataRight, experiment, params, plot_axes, ticks, titles_plot_xaxis_yaxis, datatype, marker):
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
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.yaxis.label.set_color('dodgerblue' if datatype == 'avgrunspeed' else
                             'dodgerblue' if datatype == 'runningtime' else
                             'orange' if datatype == 'idletime' else
                             'red'if datatype == 'maxspeed' else 'k')

    if experiment == 'Distance':
        data60, data90, data120 = separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, True)
        for animal in animal_list:
            if plot == "60":
                ax.plot([x+1 for x in data60[animal].keys()], [np.mean(i) for i in data60[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "90":
                ax.plot([x+1 for x in data90[animal].keys()], [np.mean(i) for i in data90[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "120":
                ax.plot([x+1 for x in data120[animal].keys()], [np.mean(i) for i in data120[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
    if experiment == 'TM_ON':
        datarev20, datarev10, datarev2, data2, data10, data20 = separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, True)
        for animal in animal_list:
            if plot == "-20":
                ax.plot([x+1 for x in datarev20[animal].keys()], [np.mean(i) for i in datarev20[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "-10":
                ax.plot([x+1 for x in datarev10[animal].keys()], [np.mean(i) for i in datarev10[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "-2":
                ax.plot([x+1 for x in datarev2[animal].keys()], [np.mean(i) for i in datarev2[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "2":
                ax.plot([x+1 for x in data2[animal].keys()], [np.mean(i) for i in data2[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "10":
                ax.plot([x+1 for x in data10[animal].keys()], [np.mean(i) for i in data10[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
            if plot == "20":
                ax.plot([x+1 for x in data20[animal].keys()], [np.mean(i) for i in data20[animal].values()], marker='o', markersize=6, color=marker[animal][0], linestyle=marker[animal][2])
    return ax


def rats_Binsession_plot(plot, animal_list, session_list, dataLeft, dataRight, experiment, params, plot_axes, ticks, titles_plot_xaxis_yaxis, datatype, marker):
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
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.yaxis.label.set_color('dodgerblue' if datatype == 'avgrunspeed' else
                             'dodgerblue' if datatype == 'runningtime' else
                             'orange' if datatype == 'idletime' else
                             'red'if datatype == 'maxspeed' else 'k')

    if experiment == 'Distance':
        data60, data90, data120 = separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, True)
        animal = plot
        ax.plot([x+1 for x in data60[animal].keys()], [np.mean(i) for i in data60[animal].values()], marker='o', markersize=6, color=marker[animal][0], linewidth=0.5, label="60", linestyle=marker[animal][2])
        ax.plot([x+1 for x in data90[animal].keys()], [np.mean(i) for i in data90[animal].values()], marker='o', markersize=6, color=marker[animal][0], linewidth=1.25, label="90", linestyle=marker[animal][2])
        ax.plot([x+1 for x in data120[animal].keys()], [np.mean(i) for i in data120[animal].values()], marker='o', markersize=6, color=marker[animal][0], linewidth=2, label="120", linestyle=marker[animal][2])
        legend_without_duplicate_labels(ax)
    if experiment == 'TM_ON':
        datarev20, datarev10, datarev2, data2, data10, data20 = separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, True)
        animal = plot
        ax.plot([x+1 for x in datarev20[animal].keys()], [np.mean(i) for i in datarev20[animal].values()], marker='o', markersize=6, color=marker[animal][0], linewidth=2, label="-20", linestyle=marker[animal][2])
        ax.plot([x+1 for x in datarev10[animal].keys()], [np.mean(i) for i in datarev10[animal].values()], marker='o', markersize=6, color=marker[animal][0], linewidth=1.5, label="-10", linestyle=marker[animal][2])
        ax.plot([x+1 for x in datarev2[animal].keys()],  [np.mean(i) for i in datarev2[animal].values()],  marker='o', markersize=6, color=marker[animal][0], linewidth=1, label="-2", linestyle=marker[animal][2])
        ax.plot([x+1 for x in data2[animal].keys()],     [np.mean(i) for i in data2[animal].values()],     marker='o', markersize=6, color=marker[animal][0], linewidth=1, label="2", linestyle=marker[animal][2])
        ax.plot([x+1 for x in data10[animal].keys()],    [np.mean(i) for i in data10[animal].values()],    marker='o', markersize=6, color=marker[animal][0], linewidth=1.5, label="10", linestyle=marker[animal][2])
        ax.plot([x+1 for x in data20[animal].keys()],    [np.mean(i) for i in data20[animal].values()],    marker='o', markersize=6, color=marker[animal][0], linewidth=2, label="20", linestyle=marker[animal][2])
        legend_without_duplicate_labels(ax)
    return ax


def rwd_plot(plot, animal_list, session_list, dataLeft, dataRight, experiment, params, plot_axes, ticks, titles_plot_xaxis_yaxis, datatype, marker):
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
    if experiment == 'TM_ON':
        ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.axvspan(-35, 50, color='grey', alpha=10/250, label="%reward: 90%", zorder=1)
    ax.axvspan(50, 135, color='grey', alpha=90/250, label="%reward: 10%", zorder=1)
    ax.yaxis.label.set_color('dodgerblue' if datatype == 'avgrunspeed' else
                             'red' if datatype == 'runningtime' else
                             'orange' if datatype == 'idletime' else
                             'red'if datatype == 'maxspeed' else 'k')
    if experiment == 'Distance':
        _, data90_90, _, _, data10_90, _ = separate_data(animal_list, session_list, dataLeft, dataRight, experiment, params, datatype, False)
        for animal in animal_list:
            if datatype == 'nb_runs':
                ax.plot((10, 90), (np.mean(data10_90[animal]), np.mean(data90_90[animal])), marker='o', markersize=6, color=marker[animal][0], zorder=2, linestyle=marker[animal][2])
            else:
                ax.plot((10, 90), (np.nanmean([item for sublist in data10_90[animal] for item in sublist]), np.nanmean([item for sublist in data90_90[animal] for item in sublist])), marker='o', markersize=6, color=marker[animal][0], zorder=2, linestyle=marker[animal][2])
    return ax


def corr_twoExps(plot, animal_list, dataLeft, dataRight, session_list1, session_list2, experiment1, experiment2, params, plot_axes, ticks, titles_plot_xaxis_yaxis, datatype, marker):
    ax = plt.gca()
    ax.set_title(titles_plot_xaxis_yaxis[0], fontsize=16, pad=20,
                 color='dodgerblue' if datatype == 'avgrunspeed' else
                       'red' if datatype == 'runningtime' else
                       'orange' if datatype == 'idletime' else
                       'red' if datatype == 'maxspeed' else 'k')
    ax.set_xlabel(titles_plot_xaxis_yaxis[1], fontsize=16)
    ax.set_ylabel(titles_plot_xaxis_yaxis[2], fontsize=16)
    ax.xaxis.label.set_color('dodgerblue' if datatype == 'avgrunspeed' else
                             'red' if datatype == 'runningtime' else
                             'orange' if datatype == 'idletime' else
                             'red'if datatype == 'maxspeed' else 'k')
    ax.yaxis.label.set_color('dodgerblue' if datatype == 'avgrunspeed' else
                             'red' if datatype == 'runningtime' else
                             'orange' if datatype == 'idletime' else
                             'red'if datatype == 'maxspeed' else 'k')
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

    xdata, ydata = {}, {}
    if experiment1 == 'Distance':
        data90_60, data90_90, data90_120, data10_60, data10_90, data10_120 = separate_data(animal_list, session_list1, dataLeft, dataRight, experiment1, params, datatype, False)
        for animal in animal_list:
            if datatype == 'nb_runs':
                a = np.mean(data90_60[animal])
                b = np.mean(data90_90[animal])
                c = np.mean(data90_120[animal])
                d = np.mean(data10_60[animal])
                e = np.mean(data10_90[animal])
                f = np.mean(data10_120[animal])
            else:
                a = np.nanmean([item for sublist in data90_60[animal] for item in sublist])
                b = np.nanmean([item for sublist in data90_90[animal] for item in sublist])
                c = np.nanmean([item for sublist in data90_120[animal] for item in sublist])
                d = np.nanmean([item for sublist in data10_60[animal] for item in sublist])
                e = np.nanmean([item for sublist in data10_90[animal] for item in sublist])
                f = np.nanmean([item for sublist in data10_120[animal] for item in sublist])
            if plot == "90%":
                xdata[animal] = (a + b + c) / 3
            if plot == "10%":
                xdata[animal] = (d + e + f) / 3
            if plot == "%":
                xdata[animal] = (d / a + e / b + f / c) / 3
                xdata[animal] = e / b

    if experiment1 == 'TM_ON':
        data90_rev20, data90_rev10, data90_rev2, data90_2, data90_10, data90_20, data10_rev20, data10_rev10, data10_rev2, data10_2, data10_10, data10_20 = separate_data(animal_list, session_list1, dataLeft, dataRight, experiment1, params, datatype, False)
        for animal in animal_list:
            if datatype == 'nb_runs':
                a = np.mean(data90_rev20[animal])
                b = np.mean(data90_rev10[animal])
                c = np.mean(data90_rev2[animal])
                d = np.mean(data90_2[animal])
                e = np.mean(data90_10[animal])
                f = np.mean(data90_20[animal])
                g = np.mean(data10_rev20[animal])
                h = np.mean(data10_rev10[animal])
                i = np.mean(data10_rev2[animal])
                j = np.mean(data10_2[animal])
                k = np.mean(data10_10[animal])
                l = np.mean(data10_20[animal])

            else:
                a = np.nanmean([item for sublist in data90_rev20[animal] for item in sublist])
                b = np.nanmean([item for sublist in data90_rev10[animal] for item in sublist])
                c = np.nanmean([item for sublist in data90_rev2[animal] for item in sublist])
                d = np.nanmean([item for sublist in data90_2[animal] for item in sublist])
                e = np.nanmean([item for sublist in data90_10[animal] for item in sublist])
                f = np.nanmean([item for sublist in data90_20[animal] for item in sublist])
                g = np.nanmean([item for sublist in data10_rev20[animal] for item in sublist])
                h = np.nanmean([item for sublist in data10_rev10[animal] for item in sublist])
                i = np.nanmean([item for sublist in data10_rev2[animal] for item in sublist])
                j = np.nanmean([item for sublist in data10_2[animal] for item in sublist])
                k = np.nanmean([item for sublist in data10_10[animal] for item in sublist])
                l = np.nanmean([item for sublist in data10_20[animal] for item in sublist])
            if plot == "90%":
                xdata[animal] = (a + b + c + d + e + f) / 6
            if plot == "10%":
                xdata[animal] = (g + h + i + j + k + l) / 6
            if plot == "%":
                xdata[animal] = (g / a + h / b + i / c + j / d + k / e + l / f) / 6

    if experiment2 == 'Distance':
        data90_60, data90_90, data90_120, data10_60, data10_90, data10_120 = separate_data(animal_list, session_list2, dataLeft, dataRight, experiment2, params, datatype, False)
        for animal in animal_list:
            if datatype == 'nb_runs':
                a = np.mean(data90_60[animal])
                b = np.mean(data90_90[animal])
                c = np.mean(data90_120[animal])
                d = np.mean(data10_60[animal])
                e = np.mean(data10_90[animal])
                f = np.mean(data10_120[animal])
            else:
                a = np.nanmean([item for sublist in data90_60[animal] for item in sublist])
                b = np.nanmean([item for sublist in data90_90[animal] for item in sublist])
                c = np.nanmean([item for sublist in data90_120[animal] for item in sublist])
                d = np.nanmean([item for sublist in data10_60[animal] for item in sublist])
                e = np.nanmean([item for sublist in data10_90[animal] for item in sublist])
                f = np.nanmean([item for sublist in data10_120[animal] for item in sublist])
            if plot == "90%":
                ydata[animal] = (a + b + c) / 3
            if plot == "10%":
                ydata[animal] = (d + e + f) / 3
            if plot == "%":
                ydata[animal] = (d / a + e / b + f / c) / 3
                ydata[animal] = e / b

    if experiment2 == 'TM_ON':
        data90_rev20, data90_rev10, data90_rev2, data90_2, data90_10, data90_20, data10_rev20, data10_rev10, data10_rev2, data10_2, data10_10, data10_20 = separate_data(animal_list, session_list2, dataLeft, dataRight, experiment2, params, datatype, False)
        for animal in animal_list:
            if datatype == 'nb_runs':
                a = np.mean(data90_rev20[animal])
                b = np.mean(data90_rev10[animal])
                c = np.mean(data90_rev2[animal])
                d = np.mean(data90_2[animal])
                e = np.mean(data90_10[animal])
                f = np.mean(data90_20[animal])
                g = np.mean(data10_rev20[animal])
                h = np.mean(data10_rev10[animal])
                i = np.mean(data10_rev2[animal])
                j = np.mean(data10_2[animal])
                k = np.mean(data10_10[animal])
                l = np.mean(data10_20[animal])
            else:
                a = np.nanmean([item for sublist in data90_rev20[animal] for item in sublist])
                b = np.nanmean([item for sublist in data90_rev10[animal] for item in sublist])
                c = np.nanmean([item for sublist in data90_rev2[animal] for item in sublist])
                d = np.nanmean([item for sublist in data90_2[animal] for item in sublist])
                e = np.nanmean([item for sublist in data90_10[animal] for item in sublist])
                f = np.nanmean([item for sublist in data90_20[animal] for item in sublist])
                g = np.nanmean([item for sublist in data10_rev20[animal] for item in sublist])
                h = np.nanmean([item for sublist in data10_rev10[animal] for item in sublist])
                i = np.nanmean([item for sublist in data10_rev2[animal] for item in sublist])
                j = np.nanmean([item for sublist in data10_2[animal] for item in sublist])
                k = np.nanmean([item for sublist in data10_10[animal] for item in sublist])
                l = np.nanmean([item for sublist in data10_20[animal] for item in sublist])
            if plot == "90%":
                ydata[animal] = (a + b + c + d + e + f) / 6
            if plot == "10%":
                ydata[animal] = (g + h + i + j + k + l) / 6
            if plot == "%":
                ydata[animal] = (g / a + h / b + i / c + j / d + k / e + l / f) / 6
    print(xdata[animal], ydata[animal])
    for animal in animal_list:
        ax.scatter(xdata[animal], ydata[animal], color=marker[animal][0])
    gradient, intercept, r_value, p_value, std_err = stats.linregress(list(xdata.values()), list(ydata.values()))
    # ax.plot((0, 200), (0, 200), alpha=0.5, color='k')
    ax.plot(np.linspace(np.min(list(xdata.values())), np.max(list(xdata.values())), 500), gradient * np.linspace(np.min(list(xdata.values())), np.max(list(xdata.values())), 500) + intercept, color='k', lw=2)
    # ax.annotate(("r=%.2f, \np=%.2f" %(r_value, p_value)), (np.max(list(xdata.values())), gradient * np.max(list(xdata.values()))+ intercept))
    return ax


"""
###------------------------------------------------------------------------------------------------------------------
### DATA PROCESSING FUNCTIONS
###------------------------------------------------------------------------------------------------------------------
"""


# Old function to compute start of run and end of run boundaries
def extract_boundaries(data, animal, session, dist, height=None):
    # animals lick in the extremities, so they spend more time there, so probability of them being there is more important than the probability of being in the middle of the apparatus. we compute the two average positions of these resting points. We defined a run as the trajectory of the animal between the resting points. So we have to find these resting points. In later stages of the experiments the start/end of runs is defined based on the speed of the animals.
    # function params : data is X position array for the session that we analyse, height = parameter to define a limit to the probability of detecting a place as significantly more probable than another.
    # We use a KDE (Kernel Density Estimate) to find said places. See testhitopeak.ipynb 2nd method for histogram. histogram is coded in next cell between """ """, but does not work on linux
    kde = stats.gaussian_kde(data)
    # compute KDE = get the position probability curve and compute peaks of the curve
    peak_pos, peak_height = [], []
    nb_samples = 120  # played a bit with the values, this works (number of data bins, we chose 1 per cm, also tested 10 bins per cm)
    samples = np.linspace(0, 120, nb_samples)
    probs = kde.evaluate(samples)
    maxima_index = find_peaks(probs, height)
    peak_pos = maxima_index[0]
    peak_height = maxima_index[1]["peak_heights"]
    # print("values", peak_pos, peak_height)
    # if there is more than two peaks (e.g. an animal decides to stay in the middle of the treadmill), only keep the two biggest peaks (should be the extremities) and remove the extra peak/s if there is one or more
    peak_posLeft, peak_heightLeft, peak_posRight, peak_heightRight = [], [], [], []
    for i, j in zip(peak_pos, peak_height):
        if i < dist/2:
            peak_posLeft.append(i)
            peak_heightLeft.append(j)
        if i > dist/2:
            peak_posRight.append(i)
            peak_heightRight.append(j)
    leftBoundaryPeak = peak_posLeft[np.argmax(peak_heightLeft)]
    rightBoundaryPeak = peak_posRight[np.argmax(peak_heightRight)]
    # print("computing bounds", animal, leftBoundaryPeak, rightBoundaryPeak)
    return leftBoundaryPeak, rightBoundaryPeak, kde


# convert scale, convert i = 0 to 120 --> 60 to-60 which correspnds to the speed to the right (0 to 60) and to the left (0 to -60)
def convert_scale(number):
    old_min = 0
    old_max = 120
    new_max = -60
    new_min = 60
    return int(((number - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)


# compute mask to separate runs and stays based on speed
def filterspeed(animal, session, dataPos, dataSpeed, dataTime, threshold, dist):
    # dissociate runs from non runs, we want a cut off based on animal, speed. How to define this speed? If we plot the speed of the animal in function of the X position in the apparatus, so we can see that there is some blobs of speeds close to 0 and near the extremities of the treadmill, these are the ones that we want to define as non running speeds. With this function we want to compute the area of these points of data (higher density, this technique might not work when animals are not properly trained) in order to differentiate them.
    middle = dist/2
    xmin, xmax = 0, 120  # specify the x and y range of the window that we want to analyse
    ymin, ymax = -60, 60
    position = np.array(dataPos, dtype=float)  # data needs to be transformed to float perform the KDE
    speed = np.array(dataSpeed, dtype=float)
    time = np.array(dataTime, dtype=float)
    X, Y = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]  # create 2D grid to compute KDE
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([position, speed])
    kernel = stats.gaussian_kde(values)  # compute KDE, this gives us an estimation of the point density
    Z = np.reshape(kernel(positions).T, X.shape)
    # Using the KDE that we just computed, we select the zones that have a density above a certain threshold (after testing 0.0001 seems to work well), which roughly corresponds to the set of data that we want to extract.
    # We loop through the 2D array, if the datapoint is > threshold we get the line (speed limit) and the row (X position in cm). This gives us the speed limit for each part of the
    # treadmill, so basically a zone delimited with speed limits (speed limits can be different in different points of the zone).
    i, j = [], []  # i is the set of speeds (lines) for which we will perform operations, j is the set of positions (rows) for each speed for which we will perform operations
    for line in range(0, len(np.rot90(Z))):
        if len(np.where(np.rot90(Z)[line] > threshold)[0]) > 1:
            i.append(convert_scale(line))
            j.append(np.where(np.rot90(Z)[line] > threshold)[0])
    # create a mask using the zone computed before and combine them. We have two zones (left and right), so we perform the steps on each side, first part is on the left.
    rawMask = np.array([])
    # pos is the array of positions for which the speed of the animal is under the speed limit. "11 [ 7  8  9 10 11]" for instance here the speed limit is 11 cm/s, and is attained between 7 and 11cm on the treadmill.
    # "10 [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 105, 106, 107, 108, 109, 110]" when we decrease the speed limit, here 10, we here have 2 zones, one between 4 and 13cm and another between 105 and 110cm.
    # we continue through these speed values (roughly from 11 to 0, then from 0 to -10 in this example)
    for line, pos in zip(i, j):
        if pos[pos < middle].size:
            low = pos[pos < middle][0]  # first value of the array, explained above (7)
            high = pos[pos < middle][-1]  # last value of the array (11)
            a = np.ma.masked_less(position, high)  # take everything left of the rightmost point, if the value is ok == True
            b = np.ma.masked_greater(position, low)  # take everything right of the leftmost point
            c = np.ma.masked_less(speed, line+0.5)  # take everything below the high point
            d = np.ma.masked_greater(speed, line-0.5)  # take everything above the low point
            mask = np.logical_and(a.mask, b.mask)  # first combination for all the rows (=Xposition), keep the intersection of mask a AND b, so keep all between the leftmost and rightmost points
            mask2 = np.logical_and(c.mask, d.mask)  # second combination for all the lines (=speed), keep the intersection of mask c AND d, so keep all between speed+0.5:speed-0.5
            combiLeft = np.logical_and(mask, mask2)  # combine the first and the second mask, so we only keep the intersection of the two masks, intersection is TRUE, the rest is FALSE
            if not rawMask.size:  # must do that for the first iteration so it's not empty
                rawMask = combiLeft
            else:
                rawMask = np.logical_xor(combiLeft, rawMask)  # merge the newly computed mask with the previously computed masks. We use XOR so that the TRUE values of the new mask are added to the complete mask. Same step left and right just add to the existing full mask wether the new mask is on the left or the right.
        # same as above for the right part
        if pos[pos > middle].size:
            low = pos[pos > middle][0]
            high = pos[pos > middle][-1]
            a = np.ma.masked_less(position, high)
            b = np.ma.masked_greater(position, low)
            c = np.ma.masked_less(speed, line + 0.5)
            d = np.ma.masked_greater(speed, line - 0.5)
            mask = np.logical_and(a.mask, b.mask)
            mask2 = np.logical_and(c.mask, d.mask)
            combiRight = np.logical_and(mask, mask2)
            if not rawMask.size:
                rawMask = combiRight
            else:
                rawMask = np.logical_xor(combiRight, rawMask)
    return ~rawMask


# Mask smoothing, what it does is if we have a small part of running in either sides between parts of not running, we say that this is not running and modify the mask. So we have to set up all possible cases and generate an appropriate response, but in all cases encountered these problems were only in the waiting times and not in running.
def removeSplits_Mask(inputMask, inputPos, animal, session, dist):
    correctedMask = [list(val) for key, val in groupby(inputMask[animal, session], lambda x: x == True)]
    splitPos = []
    middle = (dist)/2
    count = [0, 0, 0, 0, 0, 0]
    start, end = 0, 0
    for elem in correctedMask:
        start = end
        end = start + len(elem)
        splitPos.append(inputPos[animal, session][start:end])
    for m, p in zip(correctedMask, splitPos):
        if p[0] < middle and p[-1] < middle:
            if m[0] == False:
                pass
                # print("in L")
            if m[0] == True:
                # print("bug")
                correctedMask[count[5]] = [False for val in m]
                count[0] += 1
            count[5] += 1
            # print(m, p, "all left")
        elif p[0] > middle and p[-1] > middle:
            if m[0] == False:
                pass
                # print("in R")
            if m[0] == True:
                # print("bug")
                correctedMask[count[5]] = [False for val in m]
                count[1] += 1
            count[5] += 1
            # print(m, p, "all right")
        elif p[0] < middle and p[-1] > middle:
            if m[0] == True:
                pass
                # print("runLR")
            if m[0] == False:
                # print("bug")
                correctedMask[count[5]] = [True for val in m]
                count[2] += 1
            count[5] += 1
            # print(m, p, "left right")
        elif p[0] > middle and p[-1] < middle:
            if m[0] == True:
                pass
                # print("runRL")
            if m[0] == False:
                # print("bug")
                correctedMask[count[5]] = [True for val in m]
                count[3] += 1
            count[5] += 1
            # print(m, p, "right left")
        else:  # print(m, p, "bbb")
            count[4] += 1
            count[5] += 1
    # print(count)
    return np.concatenate(correctedMask)


# separate runs/stays * left/right + other variables into dicts
def extract_runSpeedBin(dataPos, dataSpeed, dataTime, dataLickR, dataLickL, openR, openL, mask, animal, session, blocks, boundary, treadmillspeed, rewardProbaBlock):
    runs = {}
    stays = {}
    runs[animal, session] = {}
    stays[animal, session] = {}
    position, speed, time, running_Xs, idle_Xs, goodSpeed, badSpeed, goodTime, badTime = ({bin: [] for bin in range(0, len(blocks))} for _ in range(9))
    speedRunToRight, speedRunToLeft, XtrackRunToRight, XtrackRunToLeft, timeRunToRight, timeRunToLeft, timeStayInRight, timeStayInLeft, XtrackStayInRight, XtrackStayInLeft, TtrackStayInRight, TtrackStayInLeft, instantSpeedRight, instantSpeedLeft, maxSpeedRight, maxSpeedLeft, whenmaxSpeedRight, whenmaxSpeedLeft, wheremaxSpeedRight, wheremaxSpeedLeft, lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft = ({bin: [] for bin in range(0, len(blocks))} for _ in range(26))
    rewardedLeft, rewardedRight = ({bin: [] for bin in range(0, len(blocks))} for _ in range(2))

    for i in range(0, len(blocks)):
        position[i] = np.array(dataPos[animal, session][i], dtype=float)
        speed[i] = np.array(dataSpeed[animal, session][i], dtype=float)
        time[i] = np.array(dataTime[animal, session][i], dtype=float)

        running_Xs[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(position[i], mask[animal, session][i])]]
        idle_Xs[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(position[i], mask[animal, session][i])]]
        goodSpeed[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(speed[i], mask[animal, session][i])]]
        badSpeed[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(speed[i], mask[animal, session][i])]]
        goodTime[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(time[i], mask[animal, session][i])]]
        badTime[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(time[i], mask[animal, session][i])]]

        stays[animal, session][i] = [[e[0], e[1], e[2], e[3], e[4]] if [e[0], e[1], e[2]] != [None, None, None] else 0 for e in [[i, j, k, l, m] for i, j, k, l, m in zip(running_Xs[i], goodSpeed[i], goodTime[i], dataLickR[animal, session][i], dataLickL[animal, session][i])]]
        runs[animal, session][i] = [[e[0], e[1], e[2], e[3], e[4]] if [e[0], e[1], e[2]] != [None, None, None] else 0 for e in [[i, j, k, l, m] for i, j, k, l, m in zip(idle_Xs[i], badSpeed[i], badTime[i], openL[i], openR[i])]]

        for run in split_a_list_at_zeros(runs[animal, session][i]):
            # calculate distance run as the distance between first and last value
            distanceRun = abs(run[0][0]-run[-1][0])
            # calculate time as sum of time interval between frames
            totaltimeRun = []
            xTrackRun = []
            instantSpeed = []
            maxSpeed = []
            valveL, valveR = [], []
            for item in run:
                xTrackRun.append(item[0])
                instantSpeed.append(abs(item[1]))  # if no abs() here > messes with maxspeed, if-TMspeed > not clean platform + TM changes direction with rat
                totaltimeRun.append(item[2])
                valveL.append(item[3])
                valveR.append(item[4])
            if np.sum(np.diff(totaltimeRun)) != 0:
                speedRun = distanceRun/np.sum(np.diff(totaltimeRun)) - treadmillspeed[i]
                maxSpeed = max(instantSpeed) - treadmillspeed[i]
                wheremaxSpeed = xTrackRun[np.argmax(instantSpeed)] - xTrackRun[0] if xTrackRun[0] < xTrackRun[np.argmax(instantSpeed)] else xTrackRun[0] - xTrackRun[np.argmax(instantSpeed)]
                whenmaxSpeed = np.sum(np.diff(totaltimeRun[0:np.argmax(instantSpeed)]))  # totaltimeRun[np.argmax(instantSpeed)]-
                # check if the subsplit starts on the left or the right -> determine if the animal is running left or right
                if run[0][0] < ((boundary[0]+boundary[1])/2):
                    # check if the subsplit is ending on the other side -> determine if this is a run
                    if run[-1][0] > ((boundary[0]+boundary[1])/2):
                        speedRunToRight[i].append(speedRun)
                        XtrackRunToRight[i].append(xTrackRun)
                        timeRunToRight[i].append(totaltimeRun)
                        instantSpeedRight[i].append(instantSpeed)
                        maxSpeedRight[i].append(maxSpeed)
                        wheremaxSpeedRight[i].append(wheremaxSpeed)
                        whenmaxSpeedRight[i].append(whenmaxSpeed)
                        if np.any(split_a_list_at_zeros(valveR)):  # if at least one != 0
                            rewardedRight[i].append(1 if split_a_list_at_zeros(valveR)[0][0] <= rewardProbaBlock[i] else 0)
                        else:
                            rewardedRight[i].append(10)

                # same thing for the runs that go to the other side
                elif run[0][0] > ((boundary[0]+boundary[1]) / 2):
                    if run[-1][0] < ((boundary[0]+boundary[1]) / 2):
                        speedRunToLeft[i].append(speedRun)
                        XtrackRunToLeft[i].append(xTrackRun)
                        timeRunToLeft[i].append(totaltimeRun)
                        instantSpeedLeft[i].append(instantSpeed)
                        maxSpeedLeft[i].append(maxSpeed)
                        wheremaxSpeedLeft[i].append(wheremaxSpeed)
                        whenmaxSpeedLeft[i].append(whenmaxSpeed)
                        if np.any(split_a_list_at_zeros(valveL)):  # if at least one != 0
                            rewardedLeft[i].append(1 if split_a_list_at_zeros(valveL)[0][0] <= rewardProbaBlock[i] else 0)
                        else:
                            rewardedLeft[i].append(10)

        for stay in split_a_list_at_zeros(stays[animal, session][i]):
            tInZone = []
            xTrackStay = []
            lickR = []
            lickL = []
            for item in stay:
                xTrackStay.append(item[0])
                tInZone.append(item[2])
                lickR.append(item[3])
                lickL.append(item[4])
            totaltimeStay = np.sum(np.diff(tInZone))
            # first identify if the subsplit created is on the left or right by comparing to the middle
            if stay[0][0] > ((boundary[0]+boundary[1]) / 2):
                # if hasLick == True:
                if not all(v == 0 for v in lickR):
                    pre = []
                    drink = []
                    post = []
                    for t, l in zip(tInZone[0:np.min(np.nonzero(lickR))], lickR[0:np.min(np.nonzero(lickR))]):
                        pre.append(t)
                    for t, l in zip(tInZone[np.min(np.nonzero(lickR)):np.max(np.nonzero(lickR))], lickR[np.min(np.nonzero(lickR)):np.max(np.nonzero(lickR))]):
                        drink.append(t)
                    for t, l in zip(tInZone[np.max(np.nonzero(lickR)):-1], lickR[np.max(np.nonzero(lickR)):-1]):
                        post.append(t)
                    # drink <- dig in that later on to have more info on lick (lick rate, number of licks, etc.)
                    lick_arrivalRight[i].append(np.sum(np.diff(pre)))
                    lick_drinkingRight[i].append(np.sum(np.diff(drink)))
                    lick_waitRight[i].append(np.sum(np.diff(post)))
                timeStayInRight[i].append(totaltimeStay)
                XtrackStayInRight[i].append(xTrackStay)
                TtrackStayInRight[i].append(tInZone)
            elif stay[0][0] < ((boundary[0] + boundary[1]) / 2):
                # if hasLick == True:
                if not all(v == 0 for v in lickL):
                    preL = []
                    drinkL = []
                    postL = []
                    for t, l in zip(tInZone[0:np.min(np.nonzero(lickL))], lickR[0:np.min(np.nonzero(lickL))]):
                        preL.append(t)
                    for t, l in zip(tInZone[np.min(np.nonzero(lickL)):np.max(np.nonzero(lickL))], lickL[np.min(np.nonzero(lickL)):np.max(np.nonzero(lickL))]):
                        drinkL.append(t)
                    for t, l in zip(tInZone[np.max(np.nonzero(lickL)):-1], lickL[np.max(np.nonzero(lickL)):-1]):
                        postL.append(t)
                    lick_arrivalLeft[i].append(np.sum(np.diff(preL)))
                    lick_drinkingLeft[i].append(np.sum(np.diff(drinkL)))
                    lick_waitLeft[i].append(np.sum(np.diff(postL)))
                timeStayInLeft[i].append(totaltimeStay)
                XtrackStayInLeft[i].append(xTrackStay)
                TtrackStayInLeft[i].append(tInZone)
    return speedRunToRight, speedRunToLeft, XtrackRunToRight, XtrackRunToLeft, timeRunToRight, timeRunToLeft, timeStayInRight, timeStayInLeft, XtrackStayInRight, XtrackStayInLeft, TtrackStayInRight, TtrackStayInLeft, instantSpeedRight, instantSpeedLeft, maxSpeedRight, maxSpeedLeft, whenmaxSpeedRight, whenmaxSpeedLeft, wheremaxSpeedRight, wheremaxSpeedLeft, lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft, rewardedRight, rewardedLeft


# due to the way blocks are computed, some runs may have started in block[n] and ended in block [n+1], this function appends the end of the run to the previous block. See reCutBins.
def fixSplittedRunsMask(animal, session, input_Binmask, blocks):
    output_Binmask = copy.deepcopy(input_Binmask)
    for i in range(1, len(blocks)):  # print(input_Binmask[i-1][-1], input_Binmask[i][0])
        if not all(v == False for v in output_Binmask[i]):  # if animal did not do any run (only one stay along the whole block) don't do the operation
            if output_Binmask[i-1][-1] == True and output_Binmask[i][0] == True:  # print(i, "case1")
                # print(session, i-1, i)  # uncomment to see whether/which bins have had fixes.
                while output_Binmask[i][0] == True:
                    output_Binmask[i-1] = np.append(output_Binmask[i-1], output_Binmask[i][0])
                    output_Binmask[i] = np.delete(output_Binmask[i], 0)
            if output_Binmask[i-1][-1] == False and output_Binmask[i][0] == False:  # print(i, "case2")
                # print(session, i-1, i)
                while output_Binmask[i][0] == False:
                    output_Binmask[i-1] = np.append(output_Binmask[i-1], output_Binmask[i][0])
                    output_Binmask[i] = np.delete(output_Binmask[i], 0)
    return output_Binmask


# following fixSplittedRunsMask, this function re cuts the bins of a variable at the same Length as the fixed binMask we just computed.
def reCutBins(data_to_cut, data_template):
    output = {}
    start_of_bin = 0
    for i, bin in enumerate(data_template):
        end_of_bin = start_of_bin + len(data_template[i])
        output[i] = data_to_cut[start_of_bin: end_of_bin]
        start_of_bin = end_of_bin
    return output


# function to stitch together all the bins of a variable to form the full session variable.
def stitch(input):
    dataSession = copy.deepcopy(input)
    for i, data in enumerate(input):
        dataSession[i] = list(itertools.chain.from_iterable(list(data.values())))
    return dataSession


#replace first 0s in animal position (animal not found / cam init) 
# if animal not found == camera edit, so replace with the first ok position
def fix_start_session(pos, edit):
    fixed = np.array(copy.deepcopy(pos))
    _edit = np.array(copy.deepcopy(edit))
    first_zero = next((i for i, x in enumerate(_edit) if not x), None)
    fixed[:first_zero] = pos[first_zero]
    _edit[:first_zero] = 0
    return fixed.flatten(), _edit.flatten()

# linear interpolation of the position when the camera did not find the animal
def fixcamglitch(time, pos, edit):
    last_good_pos = 0
    fixed = np.array(copy.deepcopy(pos))
    _ = [_ for _ in range(0, len(time))]
    _list = [[p, e, __] if e == 0 else 0 for p, e, __ in zip(pos, edit, _)]

    for i in range(1, len(_list)-1):
        if isinstance((_list[i-1]), list) and isinstance((_list[i]), int) and isinstance((_list[i+1]), list):
            _list[i] = [_list[i-1][0] + (_list[i+1][0] - _list[i-1][0])/2, 0, i]

    for _ in split_a_list_at_zeros(_list):
        if len(_) > 1:
            next_good_pos = _[0][2]
            try:
                patch = np.linspace(_list[last_good_pos][0], _list[next_good_pos][0], next_good_pos - last_good_pos + 1)
            except TypeError:
                print("TypeError, happens when restarting session in Labview whitout stopping VI, \
                cam still has last session position (so non zero and not caught by fix_start_session).\
                    Only a few cases")
            for i in range(last_good_pos, next_good_pos+1):
                fixed[i] = patch[i-last_good_pos]
            last_good_pos = _[-1][2]
    return fixed.flatten()


# DATA PROCESSING FUNCTION
def processData(arr, root, ID, sessionIN, index, buggedSessions, redoCompute=False, redoFig=False, printFigs=False, redoMask=False):
    index = index
    animal = ID

    # initialise all Var dicts
    params, rat_markers, water = {}, {}, {}
    extractTime, extractPositionX, extractPositionY, extractLickLeft, extractLickRight, framebuffer, solenoid_ON_Left, solenoid_ON_Right, cameraEdit = ({} for _ in range(9))
    rawTime, rawPositionX, rawPositionY, rawLickLeftX, rawLickRightX, rawLickLeftY, rawLickRightY, smoothMask, rawSpeed = ({} for _ in range(9))
    binPositionX, binPositionY, binTime, binLickLeftX, binLickRightX, binSolenoid_ON_Left, binSolenoid_ON_Right = ({} for _ in range(7))
    leftBoundaryPeak, rightBoundaryPeak, kde = {}, {}, {}
    smoothMask, rawMask, binSpeed, binMask = {}, {}, {}, {}
    running_Xs, idle_Xs, goodSpeed, badSpeed = {}, {}, {}, {}
    speedRunToRight,    speedRunToLeft,    XtrackRunToRight,    XtrackRunToLeft,    timeRunToRight,    timeRunToLeft,    timeStayInRight,    timeStayInLeft,    XtrackStayInRight,    XtrackStayInLeft,    TtrackStayInRight,    TtrackStayInLeft,    instantSpeedRight,    instantSpeedLeft,    maxSpeedRight,    maxSpeedLeft,    whenmaxSpeedRight,    whenmaxSpeedLeft,    wheremaxSpeedRight,    wheremaxSpeedLeft,    lick_arrivalRight,    lick_drinkingRight,    lick_waitRight,    lick_arrivalLeft,    lick_drinkingLeft,    lick_waitLeft = ({} for _ in range(26))
    speedRunToRightBin, speedRunToLeftBin, XtrackRunToRightBin, XtrackRunToLeftBin, timeRunToRightBin, timeRunToLeftBin, timeStayInRightBin, timeStayInLeftBin, XtrackStayInRightBin, XtrackStayInLeftBin, TtrackStayInRightBin, TtrackStayInLeftBin, instantSpeedRightBin, instantSpeedLeftBin, maxSpeedRightBin, maxSpeedLeftBin, whenmaxSpeedRightBin, whenmaxSpeedLeftBin, wheremaxSpeedRightBin, wheremaxSpeedLeftBin, lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin, lick_arrivalLeftBin, lick_drinkingLeftBin, lick_waitLeftBin = ({} for _ in range(26))
    nb_runs_to_rightBin, nb_runs_to_leftBin, nb_runsBin, total_trials = {}, {}, {}, {}
    nb_rewardBlockLeft, nb_rewardBlockRight, nbWaterLeft, nbWaterRight, totalWater, totalDistance = ({} for _ in range(6))
    rewardedRight, rewardedLeft, rewardedRightBin, rewardedLeftBin = {}, {}, {}, {}
    lickBug, notfixed, F00lostTRACKlick, buggedRatSessions, boundariesBug, runstaysepbug = buggedSessions


    palette = [(0.55, 0.0, 0.0),  (0.8, 0.36, 0.36),   (1.0, 0.27, 0.0),  (.5, .5, .5), (0.0, 0.39, 0.0),    (0.13, 0.55, 0.13),   (0.2, 0.8, 0.2), (.5, .5, .5)]  # we use RGB [0-1] not [0-255]. See www.colorhexa.com for conversion #old#palette = ['darkred', 'indianred', 'orangered', 'darkgreen', 'forestgreen', 'limegreen']
    if fnmatch.fnmatch(animal, 'RatF*'):
        rat_markers[animal] = [palette[index], "$\u2640$"]
    elif fnmatch.fnmatch(animal, 'RatM*'):
        rat_markers[animal] = [palette[index], "$\u2642$"]
    elif fnmatch.fnmatch(animal, 'Rat00*'):
        rat_markers[animal] = [palette[index], "$\u2426$"]
    else:
        print("error, this is not a rat you got here")

    if sessionIN != []:
        sessionList = sessionIN
    else:
        sessionList = []#sorted([os.path.basename(expPath) for expPath in glob.glob(root+os.sep+animal+os.sep+"Experiments"+os.sep+"Rat*")])
    arr[index] = 0
    time.sleep(0.1*(index+1))
    for sessionindex, session in enumerate(sessionList):
         # clear_output(wait=True)
        # update_progress(arr[:], root)
        figPath = root + os.sep + animal + os.sep + "Experiments" + os.sep + session + os.sep + "Figures" + os.sep + "recapFIG%s.png" %session
        if redoCompute == True:

            # extract/compute parameters from behav_params and create a parameter dictionnary for each rat and each session
            # change of behav_param format 07/2020 -> labview ok 27/07/2020 before nOk #format behavparam ? #catchup manual up to 27/07
            params[animal, session] = {"sessionDuration": read_params(root, animal, session, "sessionDuration"),
                                       "acqPer": read_params(root, animal, session, "acqPer"),
                                       "waterLeft": round((read_params(root, animal, session, "waterLeft", valueType=float) - read_params(root, animal, session, "cupWeight", valueType=float))/10*1000, 2),
                                       "waterRight": round((read_params(root, animal, session, "waterRight", valueType=float) - read_params(root, animal, session, "cupWeight", valueType=float))/10*1000, 2),
                                       "treadmillDist": read_params(root, animal, session, "treadmillSize"),
                                       "weight": read_params(root, animal, session, "ratWeight"),
                                       "lastWeightadlib": read_params(root, animal, session, "ratWeightadlib"),
                                       "lastDayadlib": read_params(root, animal, session, "lastDayadlib"),
                                       "lickthresholdLeft": read_params(root, animal, session, "lickthresholdLeft"),  # added in Labview 2021/07/06. Now uses the custom lickthreshold for each side. Useful when lickdata baseline drifts and value is directly changed in LV. Only one session might be bugged, so this parameter is session specific. Before, the default value (300) was used and modified manually during the analysis.
                                       "lickthresholdRight": read_params(root, animal, session, "lickthresholdRight"),
                                       "realEnd": str(read_params(root, animal, session, "ClockStop")),
                                       "brainstatus": read_params(root, animal, session, "brainstatus", valueType="other")}

            # initialize boundaries to be computed later using the KDE function
            params[animal, session]["boundaries"] = []

            # compute number of days elapsed between experiment day and removal of the water bottle
            lastDayadlib = str(datetime.datetime.strptime(str(read_params(root, animal, session, "lastDayadlib")), "%Y%m%d").date())
            stringmatch = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', session)
            experimentDay = str(datetime.datetime.strptime(stringmatch.group(), '%Y_%m_%d_%H_%M_%S'))
            daysSinceadlib = datetime.date(int(experimentDay[0:4]), int(experimentDay[5:7]), int(experimentDay[8:10])) - datetime.date(int(lastDayadlib[0:4]), int(lastDayadlib[5:7]), int(lastDayadlib[8:10]))
            params[animal, session]["daysSinceadLib"] = daysSinceadlib.days

            # compute IRL elapsed session time
            if params[animal, session]['realEnd'] != 'None':
                startExpe = datetime.time(int(experimentDay[11:13]), int(experimentDay[14:16]), int(experimentDay[17:19]))
                endExpe = datetime.time(hour=int(params[animal, session]['realEnd'][0:2]), minute=int(params[animal, session]['realEnd'][2:4]), second=int(params[animal, session]['realEnd'][4:6]))
                params[animal, session]["realSessionDuration"] = datetime.datetime.combine(datetime.date(1, 1, 1), endExpe) - datetime.datetime.combine(datetime.date(1, 1, 1), startExpe)
            else:
                params[animal, session]["realSessionDuration"] = None

            # determine block duration set based on the block timing defined in labview. 1 block in labview is comprised of a ON period and a OFF period. Max 12 blocks in LabView (12 On + 12 Off)*repeat.
            blocklist = []  # raw blocks from LabView -> 1 block (ON+OFF) + etc
            for blockN in range(1, 13):  # 13? or more ? Max 12 blocks, coded in LabView...
                # add block if  block >0 seconds then get data from file.
                # Data from behav_params as follows: Block N°: // ON block Duration // OFF block duration // Repeat block // % reward ON // % reward OFF // Treadmill speed.
                if read_params(root, animal, session, "Block " + str(blockN), dataindex=-6, valueType=str) != 0:
                    blocklist.append([read_params(root, animal, session, "Block " + str(blockN), dataindex=-6, valueType=str), read_params(root, animal, session, "Block " + str(blockN), dataindex=-5, valueType=str),
                                      read_params(root, animal, session, "Block " + str(blockN), dataindex=-4, valueType=str), read_params(root, animal, session, "Block " + str(blockN), dataindex=-3, valueType=str),
                                      read_params(root, animal, session, "Block " + str(blockN), dataindex=-2, valueType=str), read_params(root, animal, session, "Block " + str(blockN), dataindex=-1, valueType=str), blockN])
            # create an array [start_block, end_block] for each block using the values we have just read -> 1 block ON + 1 bloc OFF + etc.
            timecount, blockON_start, blockON_end, blockOFF_start, blockOFF_end = 0, 0, 0, 0, 0
            blocks = []  # blocks that we are going to use in the data processing. 1 block ON + 1 bloc OFF + etc.
            rewardP_ON = []  # probability of getting the reward in each ON phase
            rewardP_OFF = []  # same for OFF
            treadmillSpeed = []  # treadmill speed for each block (ON + OFF blocks not differenciated for now)
            rewardProbaBlock = []
            for block in blocklist:
                for repeat in range(0, block[2]):  # in essence blocks = [a, b], [b, c], [c, d], ...
                    blockON_start = timecount
                    timecount += block[0]
                    blockON_end = timecount
                    blockOFF_start = timecount
                    timecount += block[1]
                    blockOFF_end = timecount
                    blocks.append([blockON_start, blockON_end])
                    if blockOFF_start - blockOFF_end != 0:
                        blocks.append([blockOFF_start, blockOFF_end])
                    rewardP_ON.append(block[3])
                    rewardP_OFF.append(block[4])
                    rewardProbaBlock.extend(block[3:5])
                    treadmillSpeed.append(block[5])
                    treadmillSpeed.append(block[5])
            params[animal, session]["blocks"], params[animal, session]["rewardP_ON"], params[animal, session]["rewardP_OFF"], params[animal, session]["treadmillSpeed"], params[animal, session]['rewardProbaBlock'] = blocks, rewardP_ON, rewardP_OFF, treadmillSpeed, rewardProbaBlock
            
            # Extract data for each .position file generated from LabView
            # Data loaded : time array, position of the animal X and Y axis, Licks to the left and to the right, and frame number
            extractTime[animal, session]      = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[3])  # old format = 5
            extractPositionX[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[4])  # old format = 6
            extractPositionY[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[5])
            extractLickLeft[animal, session]  = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[6])
            extractLickRight[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[7])
            solenoid_ON_Left[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[8])
            solenoid_ON_Right[animal, session]= read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[9])
            framebuffer[animal, session]      = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[10])
            cameraEdit[animal, session]       = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[11])

            # Cut leftover data at the end of the session (e.g. session is 1800s long, data goes up to 1820s because session has not been stopped properly/stopped manually, so we remove the extra 20s)
            rawTime[animal, session]          =      extractTime[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawPositionX[animal, session]     = extractPositionX[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawPositionY[animal, session]     = extractPositionY[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickLeftX[animal, session]     =  extractLickLeft[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickLeftY[animal, session]     =  extractLickLeft[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]  # not needed, check
            rawLickRightX[animal, session]    = extractLickRight[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickRightY[animal, session]    = extractLickRight[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]  # not needed, check
            solenoid_ON_Left[animal, session] = solenoid_ON_Left[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            solenoid_ON_Right[animal, session]=solenoid_ON_Right[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]  # not needed, check
            cameraEdit[animal, session]       =       cameraEdit[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            
            # convert data from px to cm
            rawPositionX[animal, session], rawPositionY[animal, session] = datapx2cm(rawPositionX[animal, session]), datapx2cm(rawPositionY[animal, session])
            rawSpeed[animal, session] = compute_speed(rawPositionX[animal, session], rawTime[animal, session])
            smoothMask[animal, session] = np.array([True])

            # usually rat is not found in the first few frames, so we replace Xposition by the first nonzero value
            # this is detected as a camera edit, so we fix that as well
            rawPositionX[animal, session], cameraEdit[animal, session] = fix_start_session(rawPositionX[animal, session], cameraEdit[animal, session])
            rawPositionX[animal, session] = fixcamglitch(rawTime[animal, session], rawPositionX[animal, session], cameraEdit[animal, session])

            #######################################################################################
            # smoothing
            smoothPos, smoothSpeed = True, True
            sigmaPos, sigmaSpeed = 2, 2  # seems to work, less: not smoothed enough, more: too smoothed, not sure how to objectively compute an optimal value.
            if smoothPos == True:
                if smoothSpeed == True:
                    rawPositionX[animal, session] = smooth(rawPositionX[animal, session], sigmaPos)
                    rawSpeed[animal, session] = smooth(compute_speed(rawPositionX[animal, session], rawTime[animal, session]), sigmaSpeed)
                else:
                    rawPositionX[animal, session] = smooth(rawPositionX[animal, session], sigmaPos)
            ######################################################################################

            # Load lick data -- Licks == measure of conductance at the reward port. Conductance is ____ and when lick, increase of conductance so ___|_|___, we define it as a lick if it is above a threshold. But baseline value can randomly increase like this ___----, so baseline can be above threshold, so false detections. -> compute moving median to get the moving baseline (median, this way we eliminate the peaks in the calculation of the baseline) and then compare with threshold. __|_|__---|---|----
            window = 200
            if params[animal, session]["lickthresholdLeft"] == None:
                params[animal, session]["lickthresholdLeft"] = 300
            if params[animal, session]["lickthresholdRight"] == None:
                params[animal, session]["lickthresholdRight"] = 300
            rawLickLeftX[animal, session] = [k if i-j >= params[animal, session]["lickthresholdLeft"] else 0 for i, j, k in zip(rawLickLeftX[animal, session], movinmedian(rawLickLeftX[animal, session], window), rawPositionX[animal, session])]
            rawLickRightX[animal, session] = [k if i-j >= params[animal, session]["lickthresholdRight"] else 0 for i, j, k in zip(rawLickRightX[animal, session], movinmedian(rawLickRightX[animal, session], window), rawPositionX[animal, session])]

            # Specify if a session has lick data problems, so we don't discard the whole session (keep the run behavior, remove lick data)
            if all(v == 0 for v in rawLickLeftX[animal, session]):
                params[animal, session]["hasLick"] = False
            elif all(v == 0 for v in rawLickRightX[animal, session]):
                params[animal, session]["hasLick"] = False
            elif animal + " " + session in lickBug:
                params[animal, session]["hasLick"] = False
            else:
                params[animal, session]["hasLick"] = True

            # Water data. Drop size and volume rewarded. Compute drop size for each reward port. Determine if drops are equal, or which one is bigger. Assign properties (e.g. line width for plots) accordingly.
            limitWater_diff = 5
            watL = round(params[animal, session]["waterLeft"], 1)  # print(round(params[animal, session]["waterLeft"], 1), "µL/drop")
            watR = round(params[animal, session]["waterRight"], 1)  # print(round(params[animal, session]["waterRight"], 1), "µL/drop")
            if watL-(watL*limitWater_diff/100) <= watR <= watL+(watL*limitWater_diff/100):
                water[animal, session] = ["Same Reward Size", "Same Reward Size", 2, 2]  # print(session, "::", watL, watR, "     same L-R") #print(watL-(watL*limitWater_diff/100)) #print(watL+(watL*limitWater_diff/100))
            elif watL < watR:
                water[animal, session] = ["Small Reward", "Big Reward", 1, 5]  # print(session, "::", watL, watR, "     bigR")
            elif watL > watR:
                water[animal, session] = ["Big Reward", "Small Reward", 5, 1]  # print(session, "::", watL, watR, "     bigL")
            else:
                water[animal, session] = ["r", "r", 1, 1]

            # Compute boundaries
            border = 5  # define arbitrary border
            leftBoundaryPeak[animal, session], rightBoundaryPeak[animal, session], kde[animal, session] = extract_boundaries(rawPositionX[animal, session], animal, session, params[animal, session]['treadmillDist'], height=0.001)


            for s in boundariesBug:
                if session == s[0]:
                    params[animal, session]["boundaries"] = s[1]
                    break
                else:
                    params[animal, session]["boundaries"] = [rightBoundaryPeak[animal, session] - border, leftBoundaryPeak[animal, session] + border]
            
            # Compute or pickle run/stay mask
            maskpicklePath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session+os.sep+"Analysis"+os.sep+"mask.p"
            if os.path.exists(maskpicklePath) and (not redoMask):
                binMask[animal, session] = get_from_pickle(root, animal, session, name="mask.p")
            else:
                if session in runstaysepbug:
                    septhreshold = 0.0004
                else:
                    septhreshold = 0.0002
                rawMask[animal, session] = filterspeed(animal, session, rawPositionX[animal, session], rawSpeed[animal, session], rawTime[animal, session], septhreshold, params[animal, session]["treadmillDist"])  # threshold 0.0004 seems to work ok for all TM distances. lower the thresh the bigger the wait blob zone taken, which caused problems in 60cm configuration.
                smoothMask[animal, session] = removeSplits_Mask(rawMask, rawPositionX, animal, session, params[animal, session]["treadmillDist"])
                binMask[animal, session] = fixSplittedRunsMask(animal, session, bin_session(animal, session, smoothMask, rawTime, blocks), blocks)
            smoothMask[animal, session] = stitch([binMask[animal, session]])[0]
            running_Xs[animal, session] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(rawPositionX[animal, session], smoothMask[animal, session])]]
            idle_Xs[animal, session] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(rawPositionX[animal, session], smoothMask[animal, session])]]
            goodSpeed[animal, session] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(rawSpeed[animal, session], smoothMask[animal, session])]]
            badSpeed[animal, session] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(rawSpeed[animal, session], smoothMask[animal, session])]]
            binSpeed[animal, session] = reCutBins(rawSpeed[animal, session], binMask[animal, session])
            binTime[animal, session] = reCutBins(rawTime[animal, session], binMask[animal, session])
            binPositionX[animal, session] = reCutBins(rawPositionX[animal, session], binMask[animal, session])
            binPositionY[animal, session] = reCutBins(rawPositionY[animal, session], binMask[animal, session])
            binLickLeftX[animal, session] = reCutBins(rawLickLeftX[animal, session], binMask[animal, session])
            binLickRightX[animal, session] = reCutBins(rawLickRightX[animal, session], binMask[animal, session])
            binSolenoid_ON_Left[animal, session] = reCutBins(solenoid_ON_Left[animal, session], binMask[animal, session])
            binSolenoid_ON_Right[animal, session] = reCutBins(solenoid_ON_Right[animal, session], binMask[animal, session])

            # Extract all variables.
            speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session], XtrackRunToRightBin[animal, session], XtrackRunToLeftBin[animal, session], timeRunToRightBin[animal, session], timeRunToLeftBin[animal, session], timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session], XtrackStayInRightBin[animal, session], XtrackStayInLeftBin[animal, session], TtrackStayInRightBin[animal, session], TtrackStayInLeftBin[animal, session], instantSpeedRightBin[animal, session], instantSpeedLeftBin[animal, session], maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session], whenmaxSpeedRightBin[animal, session], whenmaxSpeedLeftBin[animal, session], wheremaxSpeedRightBin[animal, session], wheremaxSpeedLeftBin[animal, session], lick_arrivalRightBin[animal, session], lick_drinkingRightBin[animal, session], lick_waitRightBin[animal, session], lick_arrivalLeftBin[animal, session], lick_drinkingLeftBin[animal, session], lick_waitLeftBin[animal, session], rewardedRightBin[animal, session], rewardedLeftBin[animal, session] = extract_runSpeedBin(binPositionX, binSpeed, binTime, binLickRightX, binLickLeftX, binSolenoid_ON_Right[animal, session], binSolenoid_ON_Left[animal, session], binMask, animal, session, params[animal, session]['blocks'], params[animal, session]["boundaries"],  params[animal, session]["treadmillSpeed"], params[animal, session]['rewardProbaBlock'])
            speedRunToRight[animal, session],    speedRunToLeft[animal, session],    XtrackRunToRight[animal, session],    XtrackRunToLeft[animal, session],    timeRunToRight[animal, session],    timeRunToLeft[animal, session],    timeStayInRight[animal, session],    timeStayInLeft[animal, session],    XtrackStayInRight[animal, session],    XtrackStayInLeft[animal, session],    TtrackStayInRight[animal, session],    TtrackStayInLeft[animal, session],    instantSpeedRight[animal, session],    instantSpeedLeft[animal, session],    maxSpeedRight[animal, session],    maxSpeedLeft[animal, session],    whenmaxSpeedRight[animal, session],    whenmaxSpeedLeft[animal, session],    wheremaxSpeedRight[animal, session],    wheremaxSpeedLeft[animal, session],    lick_arrivalRight[animal, session],    lick_drinkingRight[animal, session],    lick_waitRight[animal, session],    lick_arrivalLeft[animal, session],    lick_drinkingLeft[animal, session],    lick_waitLeft[animal, session], rewardedRight[animal, session], rewardedLeft[animal, session] = stitch([speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session], XtrackRunToRightBin[animal, session], XtrackRunToLeftBin[animal, session], timeRunToRightBin[animal, session], timeRunToLeftBin[animal, session], timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session], XtrackStayInRightBin[animal, session], XtrackStayInLeftBin[animal, session], TtrackStayInRightBin[animal, session], TtrackStayInLeftBin[animal, session], instantSpeedRightBin[animal, session], instantSpeedLeftBin[animal, session], maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session], whenmaxSpeedRightBin[animal, session], whenmaxSpeedLeftBin[animal, session], wheremaxSpeedRightBin[animal, session], wheremaxSpeedLeftBin[animal, session], lick_arrivalRightBin[animal, session], lick_drinkingRightBin[animal, session], lick_waitRightBin[animal, session], lick_arrivalLeftBin[animal, session], lick_drinkingLeftBin[animal, session], lick_waitLeftBin[animal, session], rewardedRightBin[animal, session], rewardedLeftBin[animal, session]])
            nb_runs_to_rightBin[animal, session], nb_runs_to_leftBin[animal, session], nb_runsBin[animal, session], total_trials[animal, session] = {}, {}, {}, 0
            for i in range(0, len(params[animal, session]['blocks'])):
                nb_runs_to_rightBin[animal, session][i] = len(speedRunToRightBin[animal, session][i])
                nb_runs_to_leftBin[animal, session][i] = len(speedRunToLeftBin[animal, session][i])
                nb_runsBin[animal, session][i] = len(speedRunToRightBin[animal, session][i]) + len(speedRunToLeftBin[animal, session][i])
                total_trials[animal, session] = total_trials[animal, session] + nb_runsBin[animal, session][i]

            nb_rewardBlockLeft[animal, session], nb_rewardBlockRight[animal, session], nbWaterLeft[animal, session], nbWaterRight[animal, session] = {}, {}, 0, 0
            for i in range(0, len(params[animal, session]['blocks'])):
                nb_rewardBlockLeft[animal, session][i] = sum([1 if t[0] <= params[animal, session]['rewardProbaBlock'][i] else 0 for t in split_a_list_at_zeros(binSolenoid_ON_Left[animal, session][i])])  # split a list because in data file we have %open written along valve opening time duration (same value multiple time), so we only take the first one, verify >threshold, ...
                nb_rewardBlockRight[animal, session][i] = sum([1 if t[0] <= params[animal, session]['rewardProbaBlock'][i] else 0 for t in split_a_list_at_zeros(binSolenoid_ON_Right[animal, session][i])])  # print(i+1, nb_rewardBlockLeft[animal, session][i], nb_rewardBlockRight[animal, session][i])
            nbWaterLeft[animal, session] = sum(nb_rewardBlockLeft[animal, session].values())
            nbWaterRight[animal, session] = sum(nb_rewardBlockRight[animal, session].values())
            totalWater[animal, session] = round((nbWaterLeft[animal, session] * params[animal, session]["waterLeft"] + nbWaterRight[animal, session] * params[animal, session]["waterRight"])/1000, 2), 'mL'  # totalWater[animal, session] = nbWaterLeft[animal, session] * params[animal, session]["waterLeft"], "+", nbWaterRight[animal, session] * params[animal, session]["waterRight"]

            # compute total X distance moved during the session for each rat. maybe compute XY.
            totalDistance[animal, session] = sum(abs(np.diff(rawPositionX[animal, session])))/100

            # sequences
            changes = np.argwhere(np.diff(smoothMask[animal, session])).squeeze()
            full = []
            full.append(smoothMask[animal, session][:changes[0]+1])
            for i in range(0, len(changes)-1):
                full.append(smoothMask[animal, session][changes[i]+1:changes[i+1]+1])
            full.append(smoothMask[animal, session][changes[-1]+1:])
            fulltime = recut(rawTime[animal, session], full)
            openings = recut(solenoid_ON_Left[animal, session] + solenoid_ON_Right[animal, session], full)
            positions = recut(rawPositionX[animal, session], full)
            d = {}
            for item, (j, t, o, p) in enumerate(zip(full, fulltime, openings, positions)):
                proba = split_a_list_at_zeros(o)[0][0] if np.any(split_a_list_at_zeros(o)) else 100
                #     #action start time        #run or stay       #get reward (1) or not (0)                                                        #action duration       #dist/time=avg speed if run 
                d[item] = t[0], "run" if j[0] == True else "stay", 1 if proba < params[animal, session]['rewardProbaBlock'][get_block(t[0])] else 0, t[-1] - t[0], (p[-1] - p[0])/(t[-1] - t[0]) if j[0] == True else "wait"


        if os.path.exists(figPath) and (not redoFig):
            if printFigs == True:
                display(Image(filename=figPath))
        else:
            if redoCompute == False:
                print(session, " Error, you need to recompute everything to generate Fig.")
            else:
                # Plot figure
                fig = plt.figure(constrained_layout=False, figsize=(32, 42))
                fig.suptitle(session, y=0.9, fontsize=24)
                gs = fig.add_gridspec(75, 75)
                ax00 = fig.add_subplot(gs[0:7, 0:4])
                ax00 = plot_peak(rawPositionX[animal, session], animal, session, leftBoundaryPeak[animal, session], rightBoundaryPeak[animal, session], kde[animal, session], [0.05, 0, 0], [0, 120, 0],  marker=[""], xyLabels=["Position (cm)", "%"])
                ax01 = fig.add_subplot(gs[0:7, 5:75])
                ax01 = plot_BASEtrajectoryV2(animal, session, rawTime[animal, session], running_Xs[animal, session], idle_Xs[animal, session], rawLickLeftX[animal, session], rawLickRightX[animal, session], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration'], 50, 90, 0, 22, 10],  xyLabels=["Time (min)", " ", "Position (cm)", "", "", "", 14, 12], title=[session, "", " ", "", 16], linewidth=[1.5])
                plt.plot([0, params[animal, session]['sessionDuration']], [params[animal, session]["boundaries"][0], params[animal, session]["boundaries"][0]], ":", color='k', alpha=0.5)
                plt.plot([0, params[animal, session]['sessionDuration']], [params[animal, session]["boundaries"][1], params[animal, session]["boundaries"][1]], ":", color='k', alpha=0.5)

                gs00 = gs[8:13, 0:75].subgridspec(2, 75)
                ax11 = fig.add_subplot(gs00[0, 5:75])
                ax12 = fig.add_subplot(gs00[1, 0:75])
                ax11.plot(rawTime[animal, session], goodSpeed[animal, session], color='dodgerblue')
                ax11.plot(rawTime[animal, session], badSpeed[animal, session], color='orange')
                ax11.set_xlabel('time (s)')
                ax11.set_ylabel('speed (cm/s)')
                ax11.set_xlim(0, 3600)
                ax11.set_ylim(-200, 200)
                ax11.spines['top'].set_color("none")
                ax11.spines['right'].set_color("none")
                ax11.spines['left'].set_color("none")
                ax11.spines['bottom'].set_color("none")
                ax12.scatter(rawPositionX[animal, session], goodSpeed[animal, session], color='dodgerblue', s=0.5)
                ax12.scatter(rawPositionX[animal, session], badSpeed[animal, session], color='orange', s=0.5)
                ax12.set_xlabel('position (cm)')
                ax12.set_ylabel('speed (cm/s)')
                ax12.set_xlim(0, 130)
                ax12.set_ylim(-150, 150)
                ax12.spines['top'].set_color("none")
                ax12.spines['right'].set_color("none")
                ax12.spines['left'].set_color("none")
                ax12.spines['bottom'].set_color("none")
                yline = [0, 120]
                xline = [0, 0]
                ax12.plot(yline, xline, ":", color='k')

                ax20 = fig.add_subplot(gs[17:22, 0:10])
                ax20 = plot_tracks(animal, session, XtrackRunToRight[animal, session], timeRunToRight[animal, session], params[animal, session]["boundaries"], xylim=[-0.1, 2, 0, 120], color=['paleturquoise', 'tomato'],  xyLabels=["Time (s)", "X Position (cm)", 14], title=["Tracking run to Right",  16], linewidth=[1.5])
                ax21 = fig.add_subplot(gs[17:22, 15:25])
                ax21 = plot_tracks(animal, session, XtrackRunToLeft[animal, session], timeRunToLeft[animal, session], params[animal, session]["boundaries"], xylim=[-0.1, 2, 0, 120], color=['darkcyan', 'darkred'], xyLabels=["Time (s)", "", 14], title=["Tracking run to Left", 16], linewidth=[1.5])
                ax20 = fig.add_subplot(gs[17:22, 30:40])
                ax20 = cumul_plot(speedRunToRight[animal, session], speedRunToLeft[animal, session], barplotaxes=[0, 120, 0, 1], maxminstepbin=[0, 120, 1], scatterplotaxes=[0, 0, 0, 0], color=['paleturquoise', 'darkcyan', 'tomato', 'darkred'], xyLabels=["Speed cm/s", "Cumulative Frequency Run Speed", 14, 12], title=["Cumulative Plot Good Run Speed", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax21 = fig.add_subplot(gs[17:22, 45:55])
                ax21 = distribution_plot(speedRunToRight[animal, session], speedRunToLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 120, 1], scatterplotaxes=[0.5, 2.5, 0, 120], color=['paleturquoise', 'darkcyan', 'tomato', 'darkred'], xyLabels=["Speed (cm/s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of All Run Speed", 16], linewidth=[1.5], legend=["To Right: Good Runs ", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])

                gs23 = gs[15:22, 60:75].subgridspec(5, 2)
                ax231 = fig.add_subplot(gs23[0:2, 0:2])
                if len(framebuffer[animal, session]) != 0:
                    ax231.set_title("NbBug/TotFrames: %s/%s = %.2f" % (sum(np.diff(framebuffer[animal, session])-1), len(framebuffer[animal, session]), sum(np.diff(framebuffer[animal, session])-1)/len(framebuffer[animal, session])), fontsize=16)
                ax231.scatter(list(range(1, len(framebuffer[animal, session]))), [x-1 for x in np.diff(framebuffer[animal, session])], s=5)
                ax231.set_xlabel("frame index")
                ax231.set_ylabel("dFrame -1 (0 is ok)")
                ax232 = fig.add_subplot(gs23[3:5, 0:2])
                ax232.set_title(params[animal, session]["realSessionDuration"], fontsize=16)
                ax232.plot(np.diff(rawTime[animal, session]), label="data")
                ax232.plot(movinavg(np.diff(rawTime[animal, session]), 100), label="moving average")
                ax232.set_xlim(0, len(np.diff(rawTime[animal, session])))
                ax232.set_ylim(0, 0.1)
                ax232.set_xlabel("frame index")
                ax232.set_ylabel("time per frame (s)")

                ax30 = fig.add_subplot(gs[25:30, 0:10])
                ax30 = cumul_plot(maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 200, 0, 1], maxminstepbin=[0, 200, 1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed cm/s", "Cumulative Frequency MAX Run Speed", 14, 12], title=["Cumulative Plot MAX Run Speed", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax31 = fig.add_subplot(gs[25:30, 15:25])
                ax31 = distribution_plot(maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 200], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed (cm/s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of MAX Run Speed", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])
                ax32 = fig.add_subplot(gs[25:30, 30:40])
                ax32 = plot_speed(animal, session, instantSpeedRight[animal, session], timeRunToRight[animal, session], [0, 0], xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", "X Speed (cm/s)", 14], title=["To Right" + "\n" + "To " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", 12], linewidth=[1.5])
                ax33 = fig.add_subplot(gs[25:30, 45:55])
                ax33 = plot_speed(animal, session, instantSpeedLeft[animal, session], timeRunToLeft[animal, session], [0, 0], xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", "", 14], title=["To Left" + "\n" + "To " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", 12], linewidth=[1.5])
                ax34 = fig.add_subplot(gs[25:30, 60:70])
                ax34 = plot_speed(animal, session, instantSpeedRight[animal, session] + instantSpeedLeft[animal, session], timeRunToRight[animal, session] + timeRunToLeft[animal, session], [0, 0], xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", "", 14], title=["Speed" + "\n" + " To left and to right", 12], linewidth=[0.5])

                ax40 = fig.add_subplot(gs[35:40, 0:8])
                ax40 = cumul_plot(maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 250, 0, 1], maxminstepbin=[0, 250, 1], scatterplotaxes=[0, 0, 0, 0], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed cm/s", "Cumulative Frequency MAX Run Speed", 14, 12], title=["CumulPlt MAXrunSpeed <TreadmillCorrected>", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax41 = fig.add_subplot(gs[35:40, 12:23])
                ax41 = distribution_plot(maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 0, 0], scatterplotaxes=[0.5, 2.5, 0, 250], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed (cm/s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distr. of MAXrunSpeed <TreadmillCorrected>", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])
                ax42 = fig.add_subplot(gs[35:40, 26:34])  # where maxspeed
                ax42 = cumul_plot(wheremaxSpeedRight[animal, session], wheremaxSpeedLeft[animal, session], barplotaxes=[0, 120, 0, 1], maxminstepbin=[0, 120, 1], scatterplotaxes=[0, 0, 0, 0], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Position maxSpeed reached (cm)", "Cumulative Frequency MAX runSpeed Position", 14, 12], title=["CumulPlt MAXrunSpeed \nPosition from start of run", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax43 = fig.add_subplot(gs[35:40, 38:49])
                ax43 = distribution_plot(wheremaxSpeedRight[animal, session], wheremaxSpeedLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 0, 0], scatterplotaxes=[0.5, 2.5, 0, 120], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["X Position (cm)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distr. MAXrunSpeed \nPosition from start of run", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])
                ax44 = fig.add_subplot(gs[35:40, 52:60])  # when maxspeed
                ax44 = cumul_plot(whenmaxSpeedRight[animal, session], whenmaxSpeedLeft[animal, session], barplotaxes=[0, 2.5, 0, 1], maxminstepbin=[0, 2.5, 0.04], scatterplotaxes=[0, 0, 0, 0], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Time MAX runSpeed reached (s)", "Cumulative Frequency", 14, 12], title=["CumulPlt Time of \nMAXrunSpeed from start of run", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax45 = fig.add_subplot(gs[35:40, 64:75])
                ax45 = distribution_plot(whenmaxSpeedRight[animal, session], whenmaxSpeedLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 0, 0], scatterplotaxes=[0.5, 2.5, 0, 2.5], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Time MAX runSpeed reached (s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distr. Time of MAXrunSpeed \nfrom start of run", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])

                ax50 = fig.add_subplot(gs[45:50, 0:10])
                ax50 = plot_tracks(animal, session, XtrackStayInRight[animal, session], TtrackStayInRight[animal, session], params[animal, session]["boundaries"], xylim=[-1, 10, params[animal, session]['treadmillDist']-40, params[animal, session]['treadmillDist']], color=['moccasin', 'tomato'], xyLabels=["Time (s)", "X Position (cm)", 14, 12], title=["Tracking in Right", 16], linewidth=[1.5])
                ax51 = fig.add_subplot(gs[45:50, 15:25])
                ax51 = plot_tracks(animal, session, XtrackStayInLeft[animal, session], TtrackStayInLeft[animal, session], params[animal, session]["boundaries"], xylim=[-1, 10, 0, 40], color=['darkorange', 'darkred'], xyLabels=["Time (s)", "", 14, 12], title=["Tracking in Left", 16], linewidth=[1.5])
                ax52 = fig.add_subplot(gs[45:50, 30:40])
                ax52 = cumul_plot(timeStayInRight[animal, session], timeStayInLeft[animal, session], barplotaxes=[0, 15, 0, 1], maxminstepbin=[0, 15, 0.1], scatterplotaxes=[0, 0, 0, 0], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time in zone (s)", "Cumulative Frequency Time In Zone", 14, 12], title=["Cumulative Plot Good Time In Zone", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax53 = fig.add_subplot(gs[45:50, 45:60])
                ax53 = distribution_plot(timeStayInRight[animal, session], timeStayInLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 30, 1], scatterplotaxes=[0.5, 2.5, 0, 30], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time in zone (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of All Time In Zone", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])

                ax60 = fig.add_subplot(gs[55:60, 0:8])
                ax60 = cumul_plot(lick_arrivalRight[animal, session], lick_arrivalLeft[animal, session], barplotaxes=[0, 2, 0, 1], maxminstepbin=[0, 2, 0.1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Cumulative Frequency", 14, 12], title=["Cumulative Plot preDrink Time", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax61 = fig.add_subplot(gs[55:60, 12:23])
                ax61 = distribution_plot(lick_arrivalRight[animal, session], lick_arrivalLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 2], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution preDrink Time", 16], linewidth=[1.5], legend=["In Right", "In Left", " ", " "])
                ax62 = fig.add_subplot(gs[55:60, 26:34])
                ax62 = cumul_plot(lick_drinkingRight[animal, session], lick_drinkingLeft[animal, session], barplotaxes=[0, 4, 0, 1], maxminstepbin=[0, 4, 0.1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Cumulative Frequency", 14, 12], title=["Cumulative Plot Drink Time", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax63 = fig.add_subplot(gs[55:60, 38:49])
                ax63 = distribution_plot(lick_drinkingRight[animal, session], lick_drinkingLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 4], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of Drink Time", 16], linewidth=[1.5], legend=["In Right", "In Left", " ", " "])
                ax64 = fig.add_subplot(gs[55:60, 52:60])
                ax64 = cumul_plot(lick_waitRight[animal, session], lick_waitLeft[animal, session], barplotaxes=[0, 10, 0, 1], maxminstepbin=[0, 10, 0.1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Cumulative Frequency", 14, 12], title=["Cumulative Plot postDrink Time", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax65 = fig.add_subplot(gs[55:60, 64:75])
                ax65 = distribution_plot(lick_waitRight[animal, session], lick_waitLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 10], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of postDrink Time", 16], linewidth=[1.5], legend=["In Right", "In Left", " ", " "])

                if len(params[animal, session]['blocks']) > 1:
                    stat = "Med. "
                    ax70 = fig.add_subplot(gs[63:70, 0:9])
                    ax70 = plot_figBin([nb_runsBin[animal, session][i]/(int((blocks[i][1]-blocks[i][0])/60)) for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 25], color=['k'], xyLabels=["Time (min)", "\u0023 runs / min", 14, 12], title=["", 16], linewidth=[1.5], stat=stat)
                    ax72 = fig.add_subplot(gs[63:70, 20:29])
                    ax72 = plot_figBin([speedRunToLeftBin[animal, session][i] + speedRunToRightBin[animal, session][i] for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 100], color=['dodgerblue'], xyLabels=["Time (min)", "Avg. run speed (cm/s)", 14, 12], title=["", 16], linewidth=[1.5], scatter=True, stat=stat)
                    ax74 = fig.add_subplot(gs[63:70, 40:49])
                    ax74 = plot_figBin([maxSpeedRightBin[animal, session][i] + maxSpeedLeftBin[animal, session][i] for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 150], color=['red'], xyLabels=["Time (min)", "Average max speed (cm/s)", 14, 12], title=["", 16], linewidth=[1.5], scatter=True, stat=stat)
                    ax76 = fig.add_subplot(gs[63:70, 60:69])
                    ax76 = plot_figBin([timeStayInLeftBin[animal, session][i] + timeStayInRightBin[animal, session][i] for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 25], color=['orange'], xyLabels=["Time (min)", "Avg. time in sides (s)", 14, 12], title=["", 16], linewidth=[1.5], scatter=True, stat=stat)

                    ax71 = fig.add_subplot(gs[63:70, 10:15])
                    ax71 = plot_figBinMean(ax71, [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([nb_runsBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([nb_runsBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))], color=['k'], ylim=(0, 25))
                    ax73 = fig.add_subplot(gs[63:70, 30:35])
                    ax73 = plot_figBinMean(ax73, [np.mean(i) for i in poolByReward([speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], [np.mean(i) for i in poolByReward([speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], color=['dodgerblue'], ylim=(0, 100))
                    ax75 = fig.add_subplot(gs[63:70, 50:55])
                    ax75 = plot_figBinMean(ax75, [np.mean(i) for i in poolByReward([maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], [np.mean(i) for i in poolByReward([maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], color=['red'], ylim=(0, 150))
                    ax77 = fig.add_subplot(gs[63:70, 70:75])
                    ax77 = plot_figBinMean(ax77, [np.mean(i) for i in poolByReward([timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], [np.mean(i) for i in poolByReward([timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], color=['orange'], ylim=(0, 25))

                # %config InlineBackend.print_figure_kwargs = {'bbox_inches':None} #use % in notebook
                ax80 = fig.add_subplot(gs[73:74, 0:60])
                ax80.spines['top'].set_color("none")
                ax80.spines['right'].set_color("none")
                ax80.spines['left'].set_color("none")
                ax80.spines['bottom'].set_color("none")
                ax80.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax80.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                text = ("sessionDuration: {0} | acqPer: {1} | waterLeft: {2} | waterRight: {3} | treadmillDist: {4} | weight: {5} | lastWeightadlib: {6} | lastDayadlib: {7} | lickthresholdLeft: {8} | lickthresholdRight: {9} | realEnd: {10} | boundaries: {11} | daysSinceadLib: {12} \n realSessionDuration: {13} | blocks: {14} | \n rewardP_ON: {15} | rewardP_OFF: {16} | treadmillSpeed: {17} | rewardProbaBlock: {18} | hasLick: {19}").format(params[animal, session]['sessionDuration'], params[animal, session]['acqPer'], params[animal, session]['waterLeft'], params[animal, session]['waterRight'], params[animal, session]['treadmillDist'], params[animal, session]['weight'], params[animal, session]['lastWeightadlib'], params[animal, session]['lastDayadlib'], params[animal, session]['lickthresholdLeft'], params[animal, session]['lickthresholdRight'], params[animal, session]['realEnd'], params[animal, session]['boundaries'], params[animal, session]['daysSinceadLib'], params[animal, session]['realSessionDuration'], params[animal, session]['blocks'], params[animal, session]['rewardP_ON'], params[animal, session]['rewardP_OFF'], params[animal, session]['treadmillSpeed'], params[animal, session]['rewardProbaBlock'], params[animal, session]['hasLick'])
                ax80 = plt.text(0 ,0, str(text), wrap=True)

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### SAVE + PICKLE
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        if redoCompute == True:
            save_sessionplot_as_png(root, animal, session, 'recapFIG%s.png'%session, dpi='figure', transparent=False, background='w')
            save_as_pickle(root, params[animal, session],     animal, session, "params.p")
            save_as_pickle(root, binMask[animal, session], animal, session, "mask.p")
            save_as_pickle(root, nb_runsBin[animal, session], animal, session, "nbRuns.p")
            save_as_pickle(root, [totalDistance[animal, session], totalWater[animal, session], total_trials[animal, session]], animal, session, "misc.p")

            save_as_pickle(root, [speedRunToLeftBin[animal, session], speedRunToRightBin[animal, session]], animal, session, "avgSpeed.p")
            save_as_pickle(root, [[[np.sum(np.diff(j)) for j in timeRunToLeftBin[animal, session][i]]for i in range(0, len(params[animal, session]['blocks']))],
                                  [[np.sum(np.diff(j)) for j in timeRunToRightBin[animal, session][i]]for i in range(0, len(params[animal, session]['blocks']))]], animal, session, "timeRun.p")
            save_as_pickle(root, [maxSpeedLeftBin[animal, session], maxSpeedRightBin[animal, session]], animal, session, "maxSpeed.p")
            save_as_pickle(root, [timeStayInLeftBin[animal, session], timeStayInRightBin[animal, session]], animal, session, "timeinZone.p")
            save_as_pickle(root, [XtrackRunToLeftBin[animal, session], XtrackRunToRightBin[animal, session]], animal, session, "trackPos.p")
            save_as_pickle(root, [instantSpeedLeftBin[animal, session], instantSpeedRightBin[animal, session]], animal, session, "trackSpeed.p")
            save_as_pickle(root, [timeRunToLeftBin[animal, session], timeRunToRightBin[animal, session]], animal, session, "trackTime.p")
            save_as_pickle(root, [binLickLeftX[animal, session], binLickRightX[animal, session], binSolenoid_ON_Left[animal, session], binSolenoid_ON_Right[animal, session]], animal, session, "lick_valves.p")
            save_as_pickle(root, [rewardedRightBin[animal, session], rewardedLeftBin[animal, session]], animal, session, "rewarded.p")
            save_as_pickle(root, [TtrackStayInLeft[animal, session], TtrackStayInRight[animal, session]], animal, session, "trackTimeinZone.p")
            save_as_pickle(root, d, animal, session, "sequence.p")

            #lick_arrivalRightBin[animal, session], lick_drinkingRightBin[animal, session], lick_waitRightBin[animal, session], lick_arrivalLeftBin[animal, session], lick_drinkingLeftBin[animal, session], lick_waitLeftBin[animal, session]
            if printFigs == False:
                plt.close('all')

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### FLUSH
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # Delete all data for this session
            params, rat_markers, water = {}, {}, {}
            extractTime, extractPositionX, extractPositionY, extractLickLeft, extractLickRight, framebuffer, solenoid_ON_Left, solenoid_ON_Right, cameraEdit = ({} for i in range(9)) 
            rawTime, rawPositionX, rawPositionY, rawLickLeftX, rawLickRightX, rawLickLeftY, rawLickRightY, smoothMask, rawSpeed = ({} for i in range(9)) 
            binPositionX, binPositionY, binTime, binLickLeftX, binLickRightX, binSolenoid_ON_Left, binSolenoid_ON_Right = ({} for i in range(7))
            leftBoundaryPeak, rightBoundaryPeak, kde = {}, {}, {}
            smoothMask, rawMask, binSpeed, binMask = {}, {}, {}, {}
            running_Xs, idle_Xs, goodSpeed, badSpeed = {}, {}, {}, {}
            speedRunToRight,    speedRunToLeft,    XtrackRunToRight,    XtrackRunToLeft,    timeRunToRight,    timeRunToLeft,    timeStayInRight,    timeStayInLeft,    XtrackStayInRight,    XtrackStayInLeft,    TtrackStayInRight,    TtrackStayInLeft,    instantSpeedRight,    instantSpeedLeft,    maxSpeedRight,    maxSpeedLeft,    whenmaxSpeedRight,    whenmaxSpeedLeft,    wheremaxSpeedRight,    wheremaxSpeedLeft,    lick_arrivalRight,    lick_drinkingRight,    lick_waitRight,    lick_arrivalLeft,    lick_drinkingLeft,    lick_waitLeft    = ({} for i in range(26))
            speedRunToRightBin, speedRunToLeftBin, XtrackRunToRightBin, XtrackRunToLeftBin, timeRunToRightBin, timeRunToLeftBin, timeStayInRightBin, timeStayInLeftBin, XtrackStayInRightBin, XtrackStayInLeftBin, TtrackStayInRightBin, TtrackStayInLeftBin, instantSpeedRightBin, instantSpeedLeftBin, maxSpeedRightBin, maxSpeedLeftBin, whenmaxSpeedRightBin, whenmaxSpeedLeftBin, wheremaxSpeedRightBin, wheremaxSpeedLeftBin, lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin, lick_arrivalLeftBin, lick_drinkingLeftBin, lick_waitLeftBin = ({} for i in range(26))
            nb_runs_to_rightBin, nb_runs_to_leftBin, nb_runsBin, total_trials = {}, {}, {}, {}

        arr[index] += (1/len(sessionList))
        clear_output(wait=True)
        # update_progress(arr[:], root)


def checkHealth(arr, root, ID, sessionIN, index, buggedSessions, redoFig=False, printFigs=False, redoMask=False):
    index = index
    animal = ID 

### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### INIT
### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # initialise all Var dicts
    params, rat_markers, water = {}, {}, {}
    extractTime, extractPositionX, extractPositionY, extractLickLeft, extractLickRight, framebuffer, solenoid_ON_Left, solenoid_ON_Right, cameraEdit = ({} for i in range(9)) 
    rawTime, rawPositionX, rawPositionY, rawLickLeftX, rawLickRightX, rawLickLeftY, rawLickRightY, smoothMask, rawSpeed = ({} for i in range(9)) 
    binPositionX, binPositionY, binTime, binLickLeftX, binLickRightX, binSolenoid_ON_Left, binSolenoid_ON_Right = ({} for i in range(7))
    leftBoundaryPeak, rightBoundaryPeak, kde = {}, {}, {}
    smoothMask, rawMask, binSpeed, binMask = {}, {}, {}, {}
    running_Xs, idle_Xs, goodSpeed, badSpeed = {}, {}, {}, {}
    speedToRightCharacteristics, speedToLeftCharacteristics, speedToRightCharacteristicsBin, speedToLeftCharacteristicsBin = {}, {}, {}, {}
    limspeedRunToRight, limspeedRunToLeft, limstayRight, limstayLeft, all_speedRunToRight, all_speedRunToLeft, all_timeRunToRight, all_timeRunToLeft, all_timeStayInRight, all_timeStayInLeft, all_TtrackStayInRight, all_TtrackStayInLeft, all_instantSpeedRight, all_instantSpeedLeft, all_maxSpeedRight, all_maxSpeedLeft, good_speedRunToRight, good_speedRunToLeft, good_XtrackRunToRight, good_XtrackRunToLeft, good_timeRunToRight, good_timeRunToLeft, bad_speedRunToRight, bad_speedRunToLeft, bad_XtrackRunToRight,bad_XtrackRunToLeft, bad_timeRunToRight, bad_timeRunToLeft, good_instantSpeedRight, good_instantSpeedLeft, good_maxSpeedRight, good_maxSpeedLeft, bad_instantSpeedRight, bad_instantSpeedLeft, bad_maxSpeedRight, bad_maxSpeedLeft, good_timeStayInRight, good_timeStayInLeft, good_XtrackStayInRight, good_XtrackStayInLeft, good_TtrackStayInRight, good_TtrackStayInLeft, bad_timeStayInRight, bad_timeStayInLeft, bad_XtrackStayInRight, bad_XtrackStayInLeft, bad_TtrackStayInRight, bad_TtrackStayInLeft, lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft = ({} for i in range(54))
    limspeedRunToRightBin, limspeedRunToLeftBin, limstayRightBin, limstayLeftBin, all_speedRunToRightBin, all_speedRunToLeftBin, all_timeRunToRightBin, all_timeRunToLeftBin, all_timeStayInRightBin, all_timeStayInLeftBin, all_TtrackStayInRightBin, all_TtrackStayInLeftBin, all_instantSpeedRightBin, all_instantSpeedLeftBin, all_maxSpeedRightBin, all_maxSpeedLeftBin, good_speedRunToRightBin, good_speedRunToLeftBin, good_XtrackRunToRightBin, good_XtrackRunToLeftBin, good_timeRunToRightBin, good_timeRunToLeftBin, bad_speedRunToRightBin, bad_speedRunToLeftBin, bad_XtrackRunToRightBin, bad_XtrackRunToLeftBin, bad_timeRunToRightBin, bad_timeRunToLeftBin, good_instantSpeedRightBin, good_instantSpeedLeftBin, good_maxSpeedRightBin, good_maxSpeedLeftBin, bad_instantSpeedRightBin, bad_instantSpeedLeftBin, bad_maxSpeedRightBin, bad_maxSpeedLeftBin, good_timeStayInRightBin, good_timeStayInLeftBin, good_XtrackStayInRightBin, good_XtrackStayInLeftBin, good_TtrackStayInRightBin, good_TtrackStayInLeftBin, bad_timeStayInRightBin, bad_timeStayInLeftBin, bad_XtrackStayInRightBin, bad_XtrackStayInLeftBin, bad_TtrackStayInRightBin, bad_TtrackStayInLeftBin, lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin, lick_arrivalLeftBin, lick_drinkingLeftBin, lick_waitLeftBin = ({} for i in range(54))
    nb_runs_to_rightBin, nb_runs_to_leftBin, nb_runsBin, total_trials = {}, {}, {}, {}
    nb_rewardBlockLeft, nb_rewardBlockRight,nbWaterLeft, nbWaterRight, totalWater, totalDistance =({} for i in range(6))
    rawLickLeftplot, rawLickRightplot = {}, {}
    lickBug, notfixed, F00lostTRACKlick, buggedRatSessions, boundariesBug, runstaysepbug = buggedSessions 


    palette = [(0.4, 0.0, 0.0), (0.55, 0.0, 0.0),  (0.8, 0.36, 0.36),   (1.0, 0.27, 0.0),   (0.0, 0.39, 0.0),    (0.13, 0.55, 0.13),   (0.2, 0.8, 0.2), (0.6, 1.0, 0.6)]### we use RGB [0-1] not [0-255]. See www.colorhexa.com for conversion #old#palette = ['darkred', 'indianred', 'orangered', 'darkgreen', 'forestgreen', 'limegreen']
    if fnmatch.fnmatch(animal, 'RatF*'): rat_markers[animal]=[palette[index], "$\u2640$"]
    elif fnmatch.fnmatch(animal, 'RatM*'): rat_markers[animal]=[palette[index], "$\u2642$"]
    elif fnmatch.fnmatch(animal, 'Rat00*'): rat_markers[animal]=[palette[index], "$\u2426$"]
    else: print("error, this is not a rat you got here")

    if sessionIN != []: sessionList = sessionIN
    else: sessionList = [] #sorted([os.path.basename(expPath) for expPath in glob.glob(root+os.sep+animal+os.sep+"Experiments"+os.sep+"Rat*")])
    arr[index] = 0
    for sessionindex, session in enumerate(sessionList):
        # print(session)
        figPath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session+os.sep+"Figures" +os.sep+"healthFIG%s.png"%session
        if os.path.exists(figPath) and (not redoFig):
            if printFigs == True: display(Image(filename=figPath))
        else:

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### LOAD AND PREPROCESS DATA
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            #extract/compute parameters from behav_params and create a parameter dictionnary for each rat and each session
            #change of behav_param format 07/2020 -> labview ok 27/07/2020 before nOk #format behavparam ? #catchup manual up to 27/07

            params[animal, session] = {"sessionDuration": read_params(root, animal, session, "sessionDuration"),
                                       "acqPer": read_params(root, animal, session, "acqPer"),
                                       "waterLeft": round((read_params(root, animal, session, "waterLeft", valueType=float) - read_params(root, animal, session, "cupWeight", valueType=float))/10*1000, 2),
                                       "waterRight": round((read_params(root, animal, session, "waterRight", valueType=float) - read_params(root, animal, session, "cupWeight", valueType=float))/10*1000, 2),
                                       "treadmillDist": read_params(root, animal, session, "treadmillSize"),
                                       "weight": read_params(root, animal, session, "ratWeight"),
                                       "lastWeightadlib": read_params(root, animal, session, "ratWeightadlib"),
                                       "lastDayadlib": read_params(root, animal, session, "lastDayadlib"),
                                       "lickthresholdLeft": read_params(root, animal, session, "lickthresholdLeft"),  # added in Labview 2021/07/06. Now uses the custom lickthreshold for each side. Useful when lickdata baseline drifts and value is directly changed in LV. Only one session might be bugged, so this parameter is session specific. Before, the default value (300) was used and modified manually during the analysis.
                                       "lickthresholdRight": read_params(root, animal, session, "lickthresholdRight"),
                                       "realEnd": str(read_params(root, animal, session, "ClockStop")),
                                       "brainstatus": read_params(root, animal, session, "brainstatus", valueType="other")}


            #initialize boundaries to be computed later using the KDE function
            params[animal, session]["boundaries"] = []

            #compute number of days elapsed between experiment day and removal of the water bottle
            lastDayadlib   = str(datetime.datetime.strptime(str(read_params(root, animal, session, "lastDayadlib")), "%Y%m%d").date())
            stringmatch    = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', session)
            experimentDay  = str(datetime.datetime.strptime(stringmatch.group(), '%Y_%m_%d_%H_%M_%S'))
            daysSinceadlib = datetime.date(int(experimentDay[0:4]), int(experimentDay[5:7]), int(experimentDay[8:10])) - datetime.date(int(lastDayadlib[0:4]), int(lastDayadlib[5:7]), int(lastDayadlib[8:10]))
            params[animal, session]["daysSinceadLib"] = daysSinceadlib.days

            #compute IRL elapsed session time
            if params[animal, session]['realEnd'] != 'None':
                startExpe = datetime.time(int(experimentDay[11:13]), int(experimentDay[14:16]), int(experimentDay[17:19]))
                endExpe   = datetime.time(hour = int(params[animal, session]['realEnd'][0:2]), minute = int(params[animal, session]['realEnd'][2:4]), second = int(params[animal, session]['realEnd'][4:6]))
                params[animal, session]["realSessionDuration"] = datetime.datetime.combine(datetime.date(1, 1, 1), endExpe) - datetime.datetime.combine(datetime.date(1, 1, 1), startExpe)
            else: params[animal, session]["realSessionDuration"] = None

            #determine block duration set based on the block timing defined in labview. 1 block in labview is comprised of a ON period and a OFF period. Max 12 blocks in LabView (12 On + 12 Off)*repeat.
            blocklist = []# raw blocks from LabView -> 1 block (ON+OFF) + etc
            for blockN in range(1,13): #13? or more ? Max 12 blocks, coded in LabView...
                #add block if  block >0 seconds then get data from file. 
                #Data from behav_params as follows: Block N°: // ON block Duration // OFF block duration // Repeat block // % reward ON // % reward OFF // Treadmill speed.
                if read_params(root, animal, session, "Block "+ str(blockN), dataindex =  -6, valueType = str) != 0:
                    blocklist.append([read_params(root, animal, session, "Block "+ str(blockN), dataindex =  -6, valueType = str), read_params(root, animal, session, "Block "+ str(blockN), dataindex =  -5, valueType = str), 
                                    read_params(root, animal, session, "Block "+ str(blockN), dataindex =  -4, valueType = str), read_params(root, animal, session, "Block "+ str(blockN), dataindex =  -3, valueType = str), 
                                    read_params(root, animal, session, "Block "+ str(blockN), dataindex =  -2, valueType = str), read_params(root, animal, session, "Block "+ str(blockN), dataindex =  -1, valueType = str), blockN])
            
            #create an array [start_block, end_block] for each block using the values we have just read -> 1 block ON + 1 bloc OFF + etc.
            timecount, blockON_start, blockON_end, blockOFF_start, blockOFF_end = 0, 0, 0, 0, 0
            blocks = [] #blocks that we are going to use in the data processing. 1 block ON + 1 bloc OFF + etc.
            rewardP_ON = [] #probability of getting the reward in each ON phase 
            rewardP_OFF = [] #same for OFF
            treadmillSpeed = [] #treadmill speed for each block (ON + OFF blocks not differenciated for now)
            rewardProbaBlock = []
            for block in blocklist:
                for repeat in range(0, block[2]): #in essence blocks = [a, b], [b, c], [c, d], ...
                    blockON_start = timecount
                    timecount += block[0]
                    blockON_end = timecount
                    blockOFF_start = timecount
                    timecount += block[1]
                    blockOFF_end = timecount
                    blocks.append([blockON_start, blockON_end])
                    if blockOFF_start - blockOFF_end != 0:
                        blocks.append([blockOFF_start, blockOFF_end])
                    rewardP_ON.append(block[3])
                    rewardP_OFF.append(block[4])
                    rewardProbaBlock.extend(block[3:5])
                    treadmillSpeed.append(block[5])
                    treadmillSpeed.append(block[5])
            params[animal, session]["blocks"], params[animal, session]["rewardP_ON"], params[animal, session]["rewardP_OFF"], params[animal, session]["treadmillSpeed"], params[animal, session]['rewardProbaBlock'] = blocks, rewardP_ON, rewardP_OFF, treadmillSpeed, rewardProbaBlock
            #print(blocks)
            #Extract data for each .position file generated from LabView
            #Data loaded : time array, position of the animal X and Y axis, Licks to the left and to the right, and frame number
            extractTime[animal, session]      = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[3])#old format = 5
            extractPositionX[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[4])#old format = 6
            extractPositionY[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[5])
            extractLickLeft[animal, session]  = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[6])
            extractLickRight[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[7])
            solenoid_ON_Left[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[8])
            solenoid_ON_Right[animal, session]= read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[9])
            framebuffer[animal, session]      = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[10])
            cameraEdit[animal, session]       = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"),Col=[11])

            #Cut leftover data at the end of the session (e.g. session is 1800s long, data goes up to 1820s because session has not been stopped properly/stopped manually, so we remove the extra 20s)
            rawTime[animal, session]          = extractTime[animal, session][extractTime[animal, session]      <= params[animal, session]["sessionDuration"]]
            rawPositionX[animal, session]     = extractPositionX[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawPositionY[animal, session]     = extractPositionY[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickLeftX[animal, session]     = extractLickLeft[animal, session][extractTime[animal, session]  <= params[animal, session]["sessionDuration"]]
            rawLickLeftY[animal, session]     = extractLickLeft[animal, session][extractTime[animal, session]  <= params[animal, session]["sessionDuration"]]# not needed, check
            rawLickLeftplot[animal, session]     = extractLickLeft[animal, session][extractTime[animal, session]  <= params[animal, session]["sessionDuration"]]
            rawLickRightplot[animal, session]     = extractLickRight[animal, session][extractTime[animal, session]  <= params[animal, session]["sessionDuration"]]# not needed, check
            rawLickRightX[animal, session]    = extractLickRight[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickRightY[animal, session]    = extractLickRight[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]# not needed, check
            solenoid_ON_Left[animal, session] = solenoid_ON_Left[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            solenoid_ON_Right[animal, session]= solenoid_ON_Right[animal, session][extractTime[animal, session]<= params[animal, session]["sessionDuration"]]# not needed, check
            
            #convert data from px to cm
            rawPositionX[animal, session], rawPositionY[animal, session] = datapx2cm(rawPositionX[animal, session]), datapx2cm(rawPositionY[animal, session])
            rawSpeed[animal, session]  = compute_speed(rawPositionX[animal, session], rawTime[animal, session])
            smoothMask[animal,session] = np.array([True])

            # usually rat is not found in the first few frames, so we replace Xposition by the first nonzero value
            # this is detected as a camera edit, so we fix that as well
            rawPositionX[animal, session], cameraEdit[animal, session] = fix_start_session(rawPositionX[animal, session], cameraEdit[animal, session])
            rawPositionX[animal, session] = fixcamglitch(rawTime[animal, session], rawPositionX[animal, session], cameraEdit[animal, session])

            #smoothing
            smoothPos, smoothSpeed = True, True
            sigmaPos, sigmaSpeed = 2, 2 #seems to work, less: not smoothed enough, more: too smoothed, not sure how to objectively compute an optimal value.
            if smoothPos == True:
                if smoothSpeed == True:
                    rawPositionX[animal, session]  = smooth(rawPositionX[animal, session], sigmaPos)
                    rawSpeed[animal, session]      = smooth(compute_speed(rawPositionX[animal, session], rawTime[animal, session]), sigmaSpeed)
                else: rawPositionX[animal, session]= smooth(rawPositionX[animal, session], sigmaPos)

            # Load lick data -- Licks == measure of conductance at the reward port. Conductance is ____ and when lick, increase of conductance so ___|_|___, we define it as a lick if it is above a threshold. But baseline value can randomly increase like this ___----, so baseline can be above threshold, so false detections. -> compute moving median to get the moving baseline (median, this way we eliminate the peaks in the calculation of the baseline) and then compare with threshold. __|_|__---|---|----
            window = 200
            if params[animal, session]["lickthresholdLeft"] == None: params[animal, session]["lickthresholdLeft"] = 300
            if params[animal, session]["lickthresholdRight"]== None: params[animal, session]["lickthresholdRight"] = 300
            rawLickLeftX[animal, session]  = [k if i-j >= params[animal, session]["lickthresholdLeft"]  else 0 for i, j, k in zip(rawLickLeftX[animal, session],  reversemovinmedian(rawLickLeftX[animal, session],  window), rawPositionX[animal, session])]
            rawLickRightX[animal, session] = [k if i-j >= params[animal, session]["lickthresholdRight"] else 0 for i, j, k in zip(rawLickRightX[animal, session], reversemovinmedian(rawLickRightX[animal, session], window), rawPositionX[animal, session])]
            
            # Specify if a session has lick data problems, so we don't discard the whole session (keep the run behavior, remove lick data)     
            if   all(v == 0 for v in rawLickLeftX[animal, session]):  params[animal, session]["hasLick"] = False
            elif all(v == 0 for v in rawLickRightX[animal, session]): params[animal, session]["hasLick"] = False       
            elif animal + " " + session in lickBug: params[animal, session]["hasLick"] = False
            else: params[animal, session]["hasLick"] = True

            # Water data. Drop size and volume rewarded. Compute drop size for each reward port. Determine if drops are equal, or which one is bigger. Assign properties (e.g. line width for plots) accordingly.
            limitWater_diff = 5
            watL = round(params[animal, session]["waterLeft"], 1) #print(round(params[animal, session]["waterLeft"], 1), "µL/drop")
            watR = round(params[animal, session]["waterRight"], 1) #print(round(params[animal, session]["waterRight"], 1), "µL/drop")
            if watL-(watL*limitWater_diff/100) <= watR <= watL+(watL*limitWater_diff/100): water[animal, session] = ["Same Reward Size", "Same Reward Size", 2, 2] #print(session, "::", watL, watR, "     same L-R") #print(watL-(watL*limitWater_diff/100)) #print(watL+(watL*limitWater_diff/100))
            elif watL < watR: water[animal, session] = ["Small Reward", "Big Reward", 1, 5]#print(session, "::", watL, watR, "     bigR")
            elif watL > watR: water[animal, session] = ["Big Reward", "Small Reward", 5, 1] #print(session, "::", watL, watR, "     bigL")
            else: print(session, "error, bypass? Y/N")

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### DATA PROCESSING
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    
            # Compute boundaries
            border = 5 #define arbitrary border
            leftBoundaryPeak[animal, session], rightBoundaryPeak[animal, session], kde[animal, session] = extract_boundaries(rawPositionX[animal, session], animal, session, params[animal, session]['treadmillDist'], height = 0.001)
            # print(session, "::", leftBoundaryPeak[animal, session], rightBoundaryPeak[animal, session])
            # for s in boundariesBug: 
                # if animal + " " + session == s[0]: 
                #     print("inbug")
                #     params[animal, session]["boundaries"] = s[1]
                #     break
                # else: 
            params[animal, session]["boundaries"] = [rightBoundaryPeak[animal, session] - border, leftBoundaryPeak[animal, session] + border]


            # Compute or pickle run/stay mask
            maskpicklePath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session+os.sep+"Analysis"+os.sep+"mask.p"
            if os.path.exists(maskpicklePath) and (not redoMask): 
                binMask[animal, session] = get_from_pickle(root, animal, session, name="mask.p")
            else: 
                if animal + " " + session in runstaysepbug: 
                    septhreshold = 0.0004
                else: 
                    septhreshold = 0.0002
                # print(session, "septhreshold", septhreshold)
                rawMask[animal,session]     = filterspeed(animal, session, rawPositionX[animal, session], rawSpeed[animal, session], rawTime[animal, session], septhreshold, params[animal, session]["treadmillDist"])#threshold 0.0004 seems to work ok for all TM distances. lower the thresh the bigger the wait blob zone taken, which caused problems in 60cm configuration.
                smoothMask[animal, session] = removeSplits_Mask(rawMask, rawPositionX, animal, session, params[animal, session]["treadmillDist"])
                binMask[animal,session]     = fixSplittedRunsMask(animal, session, bin_session(animal, session, smoothMask, rawTime, blocks), blocks)
            smoothMask[animal, session] = stitch([binMask[animal, session]])[0]
            running_Xs[animal,session]     = [val[0] if val[1] == True  else None for val in [[i, j] for i, j in zip(rawPositionX[animal,session], smoothMask[animal,session])]]
            idle_Xs[animal,session]      = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(rawPositionX[animal,session], smoothMask[animal,session])]]
            goodSpeed[animal,session]   = [val[0] if val[1] == True  else None for val in [[i, j] for i, j in zip(rawSpeed[animal,session], smoothMask[animal,session])]]
            badSpeed[animal,session]    = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(rawSpeed[animal,session], smoothMask[animal,session])]]

            total_trials[animal, session] = len(split_a_list_at_zeros(stitch([binMask[animal, session]])[0])) #approximated, because we don't separate and filter runs in there, so nb of runs is deduced from the mask
            totalDistance[animal, session] = sum(abs(np.diff(rawPositionX[animal, session])))/100
            binSolenoid_ON_Left[animal, session]  = reCutBins(solenoid_ON_Left[animal, session], binMask[animal, session])
            binSolenoid_ON_Right[animal, session] = reCutBins(solenoid_ON_Right[animal, session], binMask[animal, session]) 
            
            nb_rewardBlockLeft[animal, session], nb_rewardBlockRight[animal, session],nbWaterLeft[animal, session], nbWaterRight[animal, session] = {}, {}, 0, 0
            for i in range(0, len(params[animal, session]['blocks'])):
                nb_rewardBlockLeft[animal, session][i] = sum([1 if t[0] <= params[animal, session]['rewardProbaBlock'][i] else 0 for t in split_a_list_at_zeros(binSolenoid_ON_Left[animal, session][i])]) #split a list because in data file we have %open written along valve opening time duration (same value multiple time), so we only take the first one, verify >threshold, ...
                nb_rewardBlockRight[animal, session][i] = sum([1 if t[0] <= params[animal, session]['rewardProbaBlock'][i] else 0 for t in split_a_list_at_zeros(binSolenoid_ON_Right[animal, session][i])]) #print(i+1, nb_rewardBlockLeft[animal, session][i], nb_rewardBlockRight[animal, session][i])
            nbWaterLeft[animal, session] = sum(nb_rewardBlockLeft[animal, session].values())
            nbWaterRight[animal, session] = sum(nb_rewardBlockRight[animal, session].values())
            totalWater[animal, session] = round((nbWaterLeft[animal, session] * params[animal, session]["waterLeft"] + nbWaterRight[animal, session] * params[animal, session]["waterRight"])/1000, 2), 'mL'             #totalWater[animal, session] = nbWaterLeft[animal, session] * params[animal, session]["waterLeft"], "+", nbWaterRight[animal, session] * params[animal, session]["waterRight"]

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### FIGURES
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
            
            # Plot figure
            fig = plt.figure(constrained_layout=False, figsize=(20, 16))
            fig.suptitle(session, y=0.9, fontsize=24)
            gs = fig.add_gridspec(45, 75)
            ax00 = fig.add_subplot(gs[0:7, 0:4])
            ax00 = plot_peak(rawPositionX[animal, session], animal, session, leftBoundaryPeak[animal, session], rightBoundaryPeak[animal, session], kde[animal, session],[0.05, 0, 0],[0, 120 ,0],  marker=[""], xyLabels=["Position (cm)", "%"])
            ax01 = fig.add_subplot(gs[0:7, 5:75])
            ax01 = plot_BASEtrajectoryV2(animal, session, rawTime[animal, session], running_Xs[animal, session], idle_Xs[animal, session], rawLickLeftX[animal, session], rawLickRightX[animal, session], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration'], 50, 90, 0, 22, 10],  xyLabels=["Time (min)", "", "Position (cm)", "", "", "", 14, 12], title=[session, "", "", "", 16], linewidth=[1.5])
            plt.plot([0, params[animal, session]['sessionDuration']], [params[animal, session]["boundaries"][0], params[animal, session]["boundaries"][0]], ":", color='k', alpha=0.5)
            plt.plot([0, params[animal, session]['sessionDuration']], [params[animal, session]["boundaries"][1], params[animal, session]["boundaries"][1]], ":", color='k', alpha=0.5)

            gs00 = gs[8:13, 0:75].subgridspec(2,75)
            ax11 = fig.add_subplot(gs00[0, 5:75])
            ax12 = fig.add_subplot(gs00[1, 0:75])
            ax11.plot(rawTime[animal,session], goodSpeed[animal,session], color='dodgerblue')
            ax11.plot(rawTime[animal,session], badSpeed[animal, session], color='orange')
            ax11.set_xlabel('time (s)')
            ax11.set_ylabel('speed (cm/s)')
            ax11.set_xlim(0, 3600)
            ax11.set_ylim(-200,200)
            ax11.spines['top'].set_color("none")
            ax11.spines['right'].set_color("none")
            ax11.spines['left'].set_color("none")
            ax11.spines['bottom'].set_color("none")
            ax12.scatter(rawPositionX[animal,session], goodSpeed[animal,session], color='dodgerblue', s=0.5)
            ax12.scatter(rawPositionX[animal,session], badSpeed[animal,session], color='orange', s=0.5)
            ax12.set_xlabel('position (cm)')
            ax12.set_ylabel('speed (cm/s)')
            ax12.set_xlim(0,130)
            ax12.set_ylim(-150,150)
            ax12.spines['top'].set_color("none")
            ax12.spines['right'].set_color("none")
            ax12.spines['left'].set_color("none")
            ax12.spines['bottom'].set_color("none")
            yline = [0, 120]
            xline = [0,0]
            ax12.plot(yline, xline, ":", color='k')
            
            gs23 = gs[15:20, 0:75].subgridspec(5,5)
            ax231 = fig.add_subplot(gs23[0:5, 0:2])
            if len(framebuffer[animal, session]) != 0:
                ax231.set_title("NbBug/TotFrames: %s/%s = %.2f" %(sum(np.diff(framebuffer[animal, session])-1), len(framebuffer[animal, session]), sum(np.diff(framebuffer[animal, session])-1)/len(framebuffer[animal, session])), fontsize=16)
            ax231.scatter(list(range(1, len(framebuffer[animal, session]))), [ x-1 for x in np.diff(framebuffer[animal, session])], s=5)
            ax231.set_xlabel("frame index")
            ax231.set_ylabel("dFrame -1 (0 is ok)")
            ax232 = fig.add_subplot(gs23[0:5, 3:5])
            ax232.set_title(params[animal, session]["realSessionDuration"], fontsize=16)
            ax232.plot(np.diff(rawTime[animal, session]), label="data")
            ax232.plot(movinavg(np.diff(rawTime[animal, session]), 100), label="moving average")
            ax232.set_xlim(0, len(np.diff(rawTime[animal, session])))
            ax232.set_ylim(0, 0.1)
            ax232.set_xlabel("frame index")
            ax232.set_ylabel("time per frame (s)")

            aaa = range(0, len(rawLickLeftplot[animal, session]))
            ax30 = fig.add_subplot(gs[22:27, 0:75])
            ax30.plot(rawLickLeftplot[animal, session], lw=0.5, c='k')
            ax30.plot(movinmedian(rawLickLeftplot[animal, session], window)+params[animal, session]["lickthresholdLeft"], lw=0.5, c='r')
            ax30.plot([i + params[animal, session]["lickthresholdLeft"] for i in reversemovinmedian(rawLickLeftplot[animal, session], window)], lw=1, c='g')
            ax30.scatter(aaa, [i if i-j >= params[animal, session]["lickthresholdLeft"]  else None for i, j in zip(rawLickLeftplot[animal, session],  reversemovinmedian(rawLickLeftplot[animal, session],  window))], s =10)
            ax30.set_xlabel("time")
            ax30.set_ylabel("conductance")
            ax30.set_title("LEFTLICKS, Lthrsh{0}".format(params[animal, session]["lickthresholdLeft"]), fontsize=16)
            ax30.set_xlim(0, params[animal, session]['sessionDuration']*25)
            ax30.set_ylim(0, 2000)

            ax31 = fig.add_subplot(gs[28:33, 0:75])
            ax31.plot(rawLickRightplot[animal, session], lw=0.5, c='k')
            ax31.plot(movinmedian(rawLickRightplot[animal, session], window)+params[animal, session]["lickthresholdRight"], lw=0.5, c='r')
            ax31.plot([i + params[animal, session]["lickthresholdRight"] for i in reversemovinmedian(rawLickRightplot[animal, session], window)], lw=1, c='g')
            ax31.scatter(aaa, [i if i-j >= params[animal, session]["lickthresholdRight"] else None for i, j in zip(rawLickRightplot[animal, session], reversemovinmedian(rawLickRightplot[animal, session], window))], s=10)
            ax31.set_xlabel("time")
            ax31.set_ylabel("conductance")
            ax31.set_title("RIGHTLICKS, Rthrsh{0}".format(params[animal, session]["lickthresholdRight"]), fontsize=16)
            ax31.set_xlim(0, params[animal, session]['sessionDuration']*25)
            ax31.set_ylim(0, 2000)

            ax32 = fig.add_subplot(gs[34:39, 0:75])
            ax32 = plot_BASEtrajectory(rawTime[animal, session],rawPositionX[animal, session], rawLickLeftX[animal, session], rawLickRightX[animal, session], [0, params[animal, session]['sessionDuration'], 1],[0,120,1],  color=["b", "c"], marker=["", "o", 1], linewidth=[0.5], xyLabels=["Position (px)", "Time(s)"])


            # %config InlineBackend.print_figure_kwargs={'bbox_inches':None}
            ax80 = fig.add_subplot(gs[43:45, 0:60])
            ax80.spines['top'].set_color("none")
            ax80.spines['right'].set_color("none")
            ax80.spines['left'].set_color("none")
            ax80.spines['bottom'].set_color("none")
            ax80.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
            ax80.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
            text = ("sessionDuration: {0} | acqPer: {1} | waterLeft: {2} | waterRight: {3} | treadmillDist: {4} | weight: {5} | lastWeightadlib: {6} | lastDayadlib: {7} | lickthresholdLeft: {8} | lickthresholdRight: {9} | realEnd: {10} | boundaries: {11} | daysSinceadLib: {12} \n realSessionDuration: {13} | blocks: {14} | \n rewardP_ON: {15} | rewardP_OFF: {16} | treadmillSpeed: {17} | rewardProbaBlock: {18} | hasLick: {19}").format(params[animal,session]['sessionDuration'], params[animal,session]['acqPer'], params[animal,session]['waterLeft'], params[animal,session]['waterRight'], params[animal,session]['treadmillDist'], params[animal,session]['weight'], params[animal,session]['lastWeightadlib'], params[animal,session]['lastDayadlib'], params[animal,session]['lickthresholdLeft'], params[animal,session]['lickthresholdRight'], params[animal,session]['realEnd'], params[animal,session]['boundaries'], params[animal,session]['daysSinceadLib'],params[animal,session]['realSessionDuration'], params[animal,session]['blocks'], params[animal,session]['rewardP_ON'], params[animal,session]['rewardP_OFF'], params[animal,session]['treadmillSpeed'], params[animal,session]['rewardProbaBlock'], params[animal,session]['hasLick'])
            ax80 = plt.text(0,0, str(text), wrap=True)

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### SAVE + PICKLE
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            save_sessionplot_as_png(root, animal, session, 'healthFIG%s.png'%session, dpi='figure', transparent=False, background='w')
            save_as_pickle(root, [totalDistance[animal, session], totalWater[animal, session], total_trials[animal, session]], animal, session, "misc.p")
            save_as_pickle(root, params[animal, session],     animal, session, "params.p")
            save_as_pickle(root, binMask[animal, session], animal, session, "mask.p")


            if printFigs == False:
                plt.close('all')

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### FLUSH
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
            # Delete all data for this session
            params, rat_markers, water = {}, {}, {}
            extractTime, extractPositionX, extractPositionY, extractLickLeft, extractLickRight, framebuffer, solenoid_ON_Left, solenoid_ON_Right, cameraEdit = ({} for i in range(9)) 
            rawTime, rawPositionX, rawPositionY, rawLickLeftX, rawLickRightX, rawLickLeftY, rawLickRightY, smoothMask, rawSpeed = ({} for i in range(9)) 
            binPositionX, binPositionY, binTime, binLickLeftX, binLickRightX, binSolenoid_ON_Left, binSolenoid_ON_Right = ({} for i in range(7))
            leftBoundaryPeak, rightBoundaryPeak, kde = {}, {}, {}
            smoothMask, rawMask, binSpeed, binMask = {}, {}, {}, {}
            running_Xs, idle_Xs, goodSpeed, badSpeed = {}, {}, {}, {}
            speedToRightCharacteristics, speedToLeftCharacteristics, speedToRightCharacteristicsBin, speedToLeftCharacteristicsBin = {}, {}, {}, {}
            limspeedRunToRight, limspeedRunToLeft, limstayRight, limstayLeft, all_speedRunToRight, all_speedRunToLeft, all_timeRunToRight, all_timeRunToLeft, all_timeStayInRight, all_timeStayInLeft, all_TtrackStayInRight, all_TtrackStayInLeft, all_instantSpeedRight, all_instantSpeedLeft, all_maxSpeedRight, all_maxSpeedLeft, good_speedRunToRight, good_speedRunToLeft, good_XtrackRunToRight, good_XtrackRunToLeft, good_timeRunToRight, good_timeRunToLeft, bad_speedRunToRight, bad_speedRunToLeft, bad_XtrackRunToRight,bad_XtrackRunToLeft, bad_timeRunToRight, bad_timeRunToLeft, good_instantSpeedRight, good_instantSpeedLeft, good_maxSpeedRight, good_maxSpeedLeft, bad_instantSpeedRight, bad_instantSpeedLeft, bad_maxSpeedRight, bad_maxSpeedLeft, good_timeStayInRight, good_timeStayInLeft, good_XtrackStayInRight, good_XtrackStayInLeft, good_TtrackStayInRight, good_TtrackStayInLeft, bad_timeStayInRight, bad_timeStayInLeft, bad_XtrackStayInRight, bad_XtrackStayInLeft, bad_TtrackStayInRight, bad_TtrackStayInLeft, lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft = ({} for i in range(54))
            limspeedRunToRightBin, limspeedRunToLeftBin, limstayRightBin, limstayLeftBin, all_speedRunToRightBin, all_speedRunToLeftBin, all_timeRunToRightBin, all_timeRunToLeftBin, all_timeStayInRightBin, all_timeStayInLeftBin, all_TtrackStayInRightBin, all_TtrackStayInLeftBin, all_instantSpeedRightBin, all_instantSpeedLeftBin, all_maxSpeedRightBin, all_maxSpeedLeftBin, good_speedRunToRightBin, good_speedRunToLeftBin, good_XtrackRunToRightBin, good_XtrackRunToLeftBin, good_timeRunToRightBin, good_timeRunToLeftBin, bad_speedRunToRightBin, bad_speedRunToLeftBin, bad_XtrackRunToRightBin, bad_XtrackRunToLeftBin, bad_timeRunToRightBin, bad_timeRunToLeftBin, good_instantSpeedRightBin, good_instantSpeedLeftBin, good_maxSpeedRightBin, good_maxSpeedLeftBin, bad_instantSpeedRightBin, bad_instantSpeedLeftBin, bad_maxSpeedRightBin, bad_maxSpeedLeftBin, good_timeStayInRightBin, good_timeStayInLeftBin, good_XtrackStayInRightBin, good_XtrackStayInLeftBin, good_TtrackStayInRightBin, good_TtrackStayInLeftBin, bad_timeStayInRightBin, bad_timeStayInLeftBin, bad_XtrackStayInRightBin, bad_XtrackStayInLeftBin, bad_TtrackStayInRightBin, bad_TtrackStayInLeftBin, lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin, lick_arrivalLeftBin, lick_drinkingLeftBin, lick_waitLeftBin = ({} for i in range(54))
            nb_runs_to_rightBin, nb_runs_to_leftBin, nb_runsBin, total_trials = {}, {}, {}, {}
            rawLickLeftplot, rawLickRightplot = {}, {}



##########################################################################################################################################
# Median run computation
# Modified from: Averaging GPS segments competition 2019. https://doi.org/10.1016/j.patcog.2020.107730
#                T. Karasek, "SEGPUB.IPYNB", Github 2019. https://gist.github.com/t0mk/eb640963d7d64e14d69016e5a3e93fd6
# # # should be able to squeeze SEM in SampleSet class
##########################################################################################################################################

def median(lst): 
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2    
    return sortedLst[index] 
    
def zscore(l):
    if len(np.unique(l)) == 1:
        return np.full(len(l),0.)
    return (np.array(l)  - np.mean(l)) / np.std(l)
    
def disterr(x1,y1, x2, y2):        
    sd = np.array([x1[0]-x2[0],y1[0]-y2[0]])
    ed = np.array([x1[0]-x2[-1],y1[0]-y2[-1]])
    if np.linalg.norm(sd) > np.linalg.norm(ed):
        x2 = np.flip(x2, axis=0)
        y2 = np.flip(y2, axis=0)
        
    offs = np.linspace(0,1,10)
    xrs1, yrs1 = Traj((x1,y1)).getPoints(offs)
    xrs2, yrs2 = Traj((x2,y2)).getPoints(offs)
    return np.sum(np.linalg.norm([xrs1-xrs2, yrs1-yrs2],axis=0))

def rdp(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results
    
def distance(a, b): 
    return  np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
        d = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d

class OnlyOnePointError(Exception):
    pass

class SampleSet:
    def __init__(self, ll):
        # ll is list of tuples [x_array,y_array] for every trajectory in sample
        self.trajs = [Traj(l) for l in ll]
        self.xp = None
        self.yp = None
        self.d = None
        self.filtix = None
        self.lenoutix = None
        self.disoutix = None
        self.eps = None

    def getRawAvg(self):
        trajLen = median([len(t.xs) for t in self.trajs])
        offs = np.linspace(0,1,trajLen)
        xm = []
        ym = []
        for t in self.trajs:
            xs, ys = t.getPoints(offs)
            xm.append(xs)
            ym.append(ys)        
        xp, yp = np.median(xm, axis=0), np.median(ym, axis=0)
        #xp, yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        return xp, yp

    def endpoints(self):
        cs = np.array([[self.trajs[0].xs[0],self.trajs[0].xs[-1]], [self.trajs[0].ys[0],self.trajs[0].ys[-1]]])
        xs = np.hstack([t.xs[0] for t in self.trajs] + [t.xs[-1] for t in self.trajs])
        ys = np.hstack([t.ys[0] for t in self.trajs] + [t.ys[-1] for t in self.trajs])       
        clabs = []
        oldclabs = []
        for j in range(10):
            for i in range(len(xs)):
                ap = np.array([[xs[i]],[ys[i]]])
                dists = np.linalg.norm(ap - cs, axis=0)
                clabs.append(np.argmin(dists))
            #cx = np.array([np.mean(xs[np.where(np.array(clabs)==0)]), np.mean(xs[np.where(np.array(clabs)==1)])])
            #cy = np.array([np.mean(ys[np.where(np.array(clabs)==0)]), np.mean(ys[np.where(np.array(clabs)==1)])])
            if oldclabs == clabs: 
                break
            oldclabs = clabs
            clabs = []
        for i,l in enumerate(clabs[:len(clabs)//2]):
            if l == 1:
                oldT = self.trajs[i]                
                reversedTraj = (np.flip(oldT.xs, axis=0), np.flip(oldT.ys, axis=0))
                self.trajs[i] = Traj(reversedTraj)   

    def zlen(self):
        ls = np.array([t.cuts[-1] for t in self.trajs])
        return zscore(ls)
        
    def getFiltered(self, dismax, lenlim):
        xa, ya = self.getRawAvg()
        d = zscore(np.array([disterr(t.xs, t.ys, xa, ya) for t in self.trajs]))
        l = self.zlen()
        self.lenoutix = np.where((l<lenlim[0])|(l>lenlim[1]))[0]
        lenix = np.where((l>lenlim[0])&(l<lenlim[1]))[0]
        self.disoutix = np.where(d>dismax)[0]
        disix = np.where(d<dismax)[0]
        self.d = d
        self.l = l
        self.filtix = np.intersect1d(lenix,disix)

    def getAvg(self, dismax, lenlim, eps, stat='Med.'):  # median
        self.eps = eps
        self.endpoints()        
        self.getFiltered(dismax, lenlim)
        atleast = 4
        if len(self.filtix) <= atleast:            
            distrank = np.argsort(self.d)
            self.disoutix = distrank[atleast:]
            self.lenoutix = []
            self.filtix = distrank[:atleast]
        filtered = [self.trajs[i] for i in self.filtix]
        trajLen = median([len(t.xs) for t in filtered])
        offs = np.linspace(0,1,trajLen*10)
        xm = []
        ym = []
        for t in filtered:
            xs, ys = t.getPoints(offs)            
            xm.append(xs)
            ym.append(ys)
        if stat == "Med.":
            self.xp, self.yp = zip(*rdp(list(zip(np.median(xm, axis=0),np.median(ym, axis=0))), eps))
        elif stat == "Avg.":
            self.xp, self.yp = zip(*rdp(list(zip(np.mean(xm, axis=0),np.mean(ym, axis=0))), eps))
        #self.xp, self.yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        xp, yp = self.xp,self.yp
        return xp, yp
 
    def pax(self, ax):
        ax.set_xlim(0,2.5)
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_ylim(0,130)
        for _, t in enumerate(self.trajs):    
            ax.plot(t.xs,t.ys, c="b", marker="o", markersize=2)
        for n, t in enumerate([self.trajs[i] for i in self.disoutix]):    
            ax.plot(t.xs,t.ys, c="g")
        for n, t in enumerate([self.trajs[i] for i in self.lenoutix]):    
            ax.plot(t.xs,t.ys, c="cyan")
        for n, t in enumerate([self.trajs[i] for i in np.intersect1d(self.lenoutix,self.disoutix)]):    
            ax.plot(t.xs,t.ys, c="magenta")
        if self.xp is not None:
            ax.plot(self.xp,self.yp, marker='D', color='r', linewidth=3)                

class Traj:
    def __init__(self,xsys):
        xs, ys = xsys
        a = np.array(xsys).T
        _, filtered = np.unique(a, return_index=True,axis=0)
        if len(filtered) < 2:
            raise OnlyOnePointError()
        self.xs = np.array(xs)[sorted(filtered)]
        self.ys = np.array(ys)[sorted(filtered)]
        self.xd = np.diff(xs)
        self.yd = np.diff(ys)
        self.dists = np.linalg.norm([self.xd, self.yd],axis=0)
        self.cuts = np.cumsum(self.dists)
        self.d = np.hstack([0,self.cuts])
        
    def getPoints(self, offsets):        
        offdists = offsets * self.cuts[-1]
        ix = np.searchsorted(self.cuts, offdists)        
        offdists -= self.d[ix]
        segoffs = offdists/self.dists[ix]
        x = self.xs[ix] + self.xd[ix]*segoffs
        y = self.ys[ix] + self.yd[ix]*segoffs
        return x,y     

def compute_median_trajectory(posdataRight, timedataRight, stat='Med.'):
    # eps, zmax, lenlim used in outlier detection. Here they are set so they don't exclude any outlier in the median computation. Outlying runs will be//are removed beforehand.
    eps = 0.001
    zmax = np.inf
    lenlim=(-np.inf, np.inf)
    data = list(zip([t - t[0] for t in timedataRight], posdataRight))

    ss = SampleSet(data)
    ss.getAvg(zmax, lenlim, eps, stat) # not supposed to do anything but has to be here to work ??????? Therefore, no touchy. 
    X, Y = ss.getAvg(zmax, lenlim, eps, stat)

    # Here median computation warps time (~Dynamic Time Warping) so interpolate to get back to 0.04s increments.
    interpTime = np.linspace(X[0], X[-1], int(X[-1]/0.04)+1) # create time from 0 to median arrival time, evenly spaced 0.04s
    interpPos = np.interp(interpTime, X, Y) # interpolate the position at interpTime
    return interpTime, interpPos



##########################################################################################################################################

# Reward sequence analysis functions

##########################################################################################################################################


def bin_seq(seq):
    # cut the full sequence in blocks
    prevblock = 0
    index = 0
    binseq = {k:{} for k in [_ for _ in range(0,12)]}
    for i in range(0, len(seq)): 
        if get_block(seq[i][0]) != prevblock: index = i  # if change block (next block) store action# to reset first action of next block to 0
        binseq[get_block(seq[i][0])][i-index] = seq[i]
        prevblock = get_block(seq[i][0])
    return binseq


def blank_plot(ax=None, col=None):
    if ax is None:
        ax = plt.gca()
    if col is not None:
        ax[col].axis('off') 
    else: ax.axis('off')
    return ax
    

def find_sequence(input, target_seq):
    # find the indices of the target seq in the input
    # call: find_sequence(sequence[animal, session], "0 0 1 1")
    converted = []
    for elem in range(len(input)):
        if input[elem][1] == 'run': converted.append(input[elem][2])
        else: converted.append(" ")  ######################remove that and sequence == "0001" instead of "0 0 0 1" 
    
    reward_sequence = ''.join([str(_) for _ in converted])
    max_len = len(target_seq)
    found_indices = []
    for i in range(len(reward_sequence)):
        chunk = reward_sequence[i:i+max_len+1]
        for j in range(1, len(chunk)+1):
            seq = chunk[:j]
            if seq == target_seq:
                #if i>10 and i<len(reward_sequence)-15:
                    found_indices.append(i+len(seq)-1)
    return found_indices


def plot_around_indices(input, target_seq, var, ax=None):
    # plot variable during and after input sequence
    # call: plot_around_indices(sequence[animal, session], "0 0 1 1", "speed", axs[0])
    indices = find_sequence(input, target_seq)
    print(f'Found {len(indices)} matches')
    if ax is None: ax = plt.gca()
    ax.plot((0, 0), (-1, 100), c='k', ls='--', lw=2)

    if len(indices) > 0:

        yy = np.empty((len(indices), 22))
        x = np.arange(-10, 11, 1.0)[::2]
        meanvar = []

        for i, index in enumerate(indices):
            
            if index > 10: low = index-10
            else: low = 0

            if index > len(input)-15: high = len(input)-1
            else: high = index+11 

            if var == "speed":
                for r in range(low, high):
                    if input[r][1] == 'run':
                        ax.scatter(np.random.normal(r-index, 0.25, 1), abs(input[r][4]), c='dodgerblue', s=4)
                        try: yy[i][r-index+10] = abs(input[r][4])
                        except IndexError: pass
                for r in range(index+1, high):
                    if input[r][1] == 'run':
                        meanvar.append(abs(input[r][4]))
                y = np.median(yy, axis=0)[::2]
                ax.set_ylim(20, 120)
                
            if var == "wait":
                for r in range(low, high+1):
                    if input[r][1] == 'stay':
                        ax.scatter(np.random.normal(r-index-1, 0.25, 1), input[r][3], c='orange', s=4)
                        try: yy[i][r-index+10] = input[r][3]
                        except IndexError: pass
                for r in range(index+2, high+1):
                    if input[r][1] == 'stay':
                        meanvar.append(input[r][3])
                y = np.median(yy, axis=0)[1::2]
                ax.set_ylim(0, 8)

            if var == "run":
                for r in range(low, high):
                    if input[r][1] == 'run':
                        ax.scatter(np.random.normal(r-index, 0.25, 1), input[r][3], c='red', s=4)
                        try: yy[i][r-index+10] = input[r][3]
                        except IndexError: pass
                for r in range(index+1, high):
                    if input[r][1] == 'run':
                        meanvar.append(abs(input[r][3]))
                y = np.median(yy, axis=0)[::2]
                ax.set_ylim(0, 4)

        # axvspan previous runs rewarded
        a, b = 0, 2
        m = len(target_seq)+1
        _ = [-i for i in range(-1, m)][::-1]
        for yn in target_seq:
            if yn != " ": ax.axvspan(_[a], _[b], color='g' if yn == "1" else 'r', alpha=0.25,)
            a += 1
            b += 1


        # average of next 5 items
        ax.scatter(np.random.normal(13, 0.25, len(meanvar)), meanvar, c='k', s=4)
        ax.scatter(13, np.nanmedian(meanvar), c='gray', s=50)
        print(f'{np.nanmedian(meanvar) :.2f} ± {stats.sem(meanvar):.2f}')
        ax.plot(x, y, c='k', marker='o')  

    ax.set_xticks(np.arange(-10, 11, 2.0))
    ax.set_xticklabels(["-5#", "-4#", "-3#", "-2#", "-1#", 0, "+1#", "+2#", "+3#", "+4#", "+5#"])
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    return ax


def compute_around_indices(input, target_seq, var, next=1, verbose=False):
    # compute median of following items
    # call: compute_around_indices(binseq[animal, session][11], "1", "speed", next=1)
    indices = find_sequence(input, target_seq)
    if verbose: print(f'Found {len(indices)} matches')

    if len(indices) > 0:
        meanvar = []

        for i, index in enumerate(indices):
            
            if index > len(input)-(next*3): high = len(input)-2
            else: high = index+(2*next)

            if var == "speed":
                for r in range(index+1, high+1):
                    if input[r][1] == 'run':
                        meanvar.append(abs(input[r][4]))
                        # print(r, index, abs(input[r][4]))
                
            if var == "wait":
                for r in range(index, high):
                    if input[r][1] == 'stay':
                        meanvar.append(input[r][3])

            if var == "run":
                for r in range(index+1, high+1):
                    if input[r][1] == 'run':
                        meanvar.append(abs(input[r][3]))
        if verbose: print(f'{np.nanmedian(meanvar) :.2f} ± {stats.sem(meanvar):.2f}')
        return meanvar
    else: return None


def plotseq2(axs, input, targetlist, var, dat, next=1, axes=[0, 50, 0, 120]):
    if dat == 'blockbyblock': nbbins = 12
    elif dat == 'firsttwo': nbbins = 2
    elif dat == 'pooled': nbbins = 2
    
    for i in range(nbbins):
        data = []
        res = dict()
        median, error, df, lw = dict(), dict(), dict(), dict()
        if dat == 'pooled':
            for elem in range(len(input)):
                if 0+i*300 < input[elem][0] <=300+i*300:
                    data.append(input[elem])
                elif 600+i*300 < input[elem][0] <=900+i*300:
                    data.append(input[elem])
                elif 1200+i*300 < input[elem][0] <=1500+i*300:
                    data.append(input[elem])
                elif 1800+i*300 < input[elem][0] <=2100+i*300:
                    data.append(input[elem])
                elif 2400+i*300 < input[elem][0] <=2700+i*300:
                    data.append(input[elem])
                elif 3000+i*300 < input[elem][0] <=3300+i*300:
                    data.append(input[elem])
                else: pass
        else:
            for elem in range(len(input)):
                if i*300 < input[elem][0] <= (i+1)*300:
                    data.append(input[elem])
                else: pass

        for index, target in enumerate(targetlist):
            sol = compute_around_indices(data, target, var, next=next)
            if sol is not None:
                prop = int(np.mean([int(i) for i in target.split()])*100)
                if prop in res.keys(): res[prop].extend(sol)
                if prop not in res.keys(): res[prop] = sol

        for key in res.keys():
            lw[key] = 1
            median[key], error[key], df[key] = np.nanmedian(res[key]), stats.sem(res[key]), len(res[key])
            # axs[i].annotate(f'{key}: {median[key]:.2f} ± {error[key]:.2f}', (0, key/100))

        c = 1.3
        for key1, key2 in itertools.combinations(median.keys(), 2):
            t_stat, dff, cv, p = _ttest_wprecomputed(median[key1], median[key2], error[key1], error[key2], df[key1], df[key2], alpha=0.05)
            axs[i].annotate(f'{key1}-{key2}: \n t={t_stat:.2f}, p={p:.2f}', (0, c))
            if p <= 0.05: 
                
                # axs[i].annotate(f'{key1}-{key2}: \n t={t_stat:.2f}, df={dff:d}, cv={cv:.2f}, p={p:.2f}', (0, c))
                c-=.2
                if median[key1] > median[key2]: lw[key1] += 1
                else: lw[key2] += 1
                # print(f'block {i} {key1}-{key2}: {median[key1]:.1f}±{error[key1]:.1f}//{median[key2]:.1f}±{error[key2]:.1f} :: t={t_stat:.2f}, df={dff:d}, cv={cv:.2f}, p={p:.2f}')

        for key in res.keys():
            axs[i].hist(res[key], np.arange(-1, axes[1]+1, 1), 
                weights=np.ones_like(res[key])/float(len(res[key])), 
                color=(key/100, 0, 0), histtype='step', linewidth=lw[key], 
                label = f'{key}% R over {len([int(i) for i in target.split()])} prev. runs',
                cumulative=True,
                )

        alph = 90 if i%2==0 else 10
        axs[i].axvspan(-1, axes[1], color='gray', alpha=alph/250, zorder=1)
        axs[i].set_ylim(-.1, 1.8)
        axs[i].set_xlim(-1, axes[1])
        axs[i].legend()
    axs[0].set_ylabel("cumul")
    axs[0].set_xlabel(var)
    return axs


def plotseq3(axs, input, targetlist, var, dat, next=1, axes=[0, 50, 0, 120]):
    if dat == 'blockbyblock': nbbins = 12
    elif dat == 'firsttwo': nbbins = 2
    elif dat == 'pooled': nbbins = 2
    elif dat == 'mixed': nbbins = 1
    
    for i in range(nbbins):
        data = []
        res = dict()
        median, error, df, lw = dict(), dict(), dict(), dict()
        if dat == 'pooled':
            for elem in range(len(input)):
                if 0+i*300 < input[elem][0] <=300+i*300:
                    data.append(input[elem])
                elif 600+i*300 < input[elem][0] <=900+i*300:
                    data.append(input[elem])
                elif 1200+i*300 < input[elem][0] <=1500+i*300:
                    data.append(input[elem])
                elif 1800+i*300 < input[elem][0] <=2100+i*300:
                    data.append(input[elem])
                elif 2400+i*300 < input[elem][0] <=2700+i*300:
                    data.append(input[elem])
                elif 3000+i*300 < input[elem][0] <=3300+i*300:
                    data.append(input[elem])
                else: pass

        elif dat == 'mixed':
            data = input
        else:
            for elem in range(len(input)):
                if i*300 < input[elem][0] <= (i+1)*300:
                    data.append(input[elem])
                else: pass

        for index, target in enumerate(targetlist):
            sol = compute_around_indices(data, target, var, next=next)
            if sol is not None:
                prop = int(np.mean([int(i) for i in target.split()])*100)
                if prop in res.keys(): res[prop].extend(sol)
                if prop not in res.keys(): res[prop] = sol

        # if i==0: 
        #     for key in res.keys():
        #         print(key, res[key])

        for key in res.keys():
            lw[key] = 1
            median[key], error[key], df[key] = np.nanmedian(res[key]), stats.sem(res[key]), len(res[key])
            # axs[i].annotate(f'{key}: {median[key]:.2f} ± {error[key]:.2f}', (0, key/100))

        # # anova
        # F, p = stats.f_oneway(*res.values())
        # axs[i].annotate(f'{F:.2f}, p={p:.2f}', (0, 1.3))

        # # regression points
        # x = np.concatenate([[key/100 for _ in res[key]] for key in res.keys()])
        # y = np.concatenate([val for val in res.values()])
        # #regression medians
        # x = [key/100 for key in res.keys()]
        # y = [np.median(val) for val in res.values()]

        # gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # axs[i].plot(np.linspace(np.min(x), np.max(x), 500), gradient * np.linspace(np.min(x), np.max(x), 500) + intercept, color='k' if p_value > 0.05 else 'r', lw=2.5)
        # axs[i].annotate(f'{gradient:.2f}, p={p_value:.2f}', (0, 1.3))

        # t test
        c = 1.3
        for key1, key2 in itertools.combinations(median.keys(), 2):
            t_stat, dff, cv, p = _ttest_wprecomputed(median[key1], median[key2], error[key1], error[key2], df[key1], df[key2], alpha=0.05)
            if p <= 0.05: 
                # axs[i].annotate(f'{key1}-{key2}: \n t={t_stat:.2f}, df={dff:d}, cv={cv:.2f}, p={p:.2f}', (0, c))
                c-=.2
                if median[key1] > median[key2]: lw[key1] += 1
                else: lw[key2] += 1
                # print(f'block {i} {key1}-{key2}: {median[key1]:.1f}±{error[key1]:.1f}//{median[key2]:.1f}±{error[key2]:.1f} :: t={t_stat:.2f}, df={dff:d}, cv={cv:.2f}, p={p:.2f}')

        for key in res.keys():
            axs[i].errorbar(key/100, median[key], yerr=error[key], capsize=0, color='black', elinewidth=5, zorder=2)
            axs[i].scatter(key/100, median[key], marker='o', color='red', s=30, zorder=3)
            axs[i].scatter([np.random.normal(key/100, 0.05, len(res[key]))], res[key], s=5);
            axs[i].violinplot(res[key], [key/100])
            # axs[i].annotate(f'{median[key]:.2f}±{error[key]:.2f}', ((key/100)-.25, median[key]-(median[key]/10)))

        alph = 90 if i%2==0 else 10
        axs[i].axvspan(-1, axes[1], color='gray', alpha=alph/250, zorder=1)
        axs[i].set_ylim(axes[2], axes[3])
        axs[i].set_xlim(axes[0], axes[1])
        # axs[i].legend()
    axs[0].set_xlabel(f'Rwd over {len([int(i) for i in target.split()])} prev. runs')
    axs[0].set_ylabel(var)
    return axs


def pool_sequences(animal, sessionlist, input):
    # combine sequences for multiple sessions
    # call: pool_sequences(animal, matchsession(animal, dist120), sequence)
    output = dict()
    c = 0
    for session in sessionlist:
        for elem in range(len(input[animal, session])):
            output[c] = input[animal, session][elem]
            c += 1
        for _ in range(15):
            output[c] = (-1, -1, -1, -1, -1)
            c += 1
    return output


def _generate_targetList(seq_len=1):
    # generate list of all reward combinations for specified sequence length
    # call: generate_targetList(seq_len=2)
    get_binary = lambda x: format(x, 'b')
    output = []
    for i in range(2**seq_len):
        # list binary number from 0 to 2**n, add leading zeroes when resulting seq is too short 
        binstr = "0" * abs(len(get_binary(i)) - seq_len) + str(get_binary(i))
        output.append(binstr.replace("", " ")[1: -1])
    return output


def _ttest_wprecomputed(mean1, mean2, se1, se2, df1, df2, alpha=0.05):
    sed = np.sqrt(se1**2.0 + se2**2.0)
    t_stat = (mean1 - mean2) / sed
    df = df1 + df2 - 2
    cv = stats.t.ppf(1.0 - alpha, df)  # critical value
    p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
    #print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
    return t_stat, df, cv, p


# block by block
def plotseqinter(axs, input, targetlist, var, dat, next=1, axes=[0, 50, 0, 120]):
    for i in range(12):
        data = []
        inter = []
        _seq = np.empty([len(input), 0]).tolist()
        resblock = dict()
        resinter = dict()
        medianblock, errorblock, dfblock, lwblock = dict(), dict(), dict(), dict()
        medianinter, errorinter, dfinter, lwinter = dict(), dict(), dict(), dict()

        for elem in range(2*targetlist, len(input)):
            for seqitem in range(elem - 2*targetlist, elem):
                _seq[elem].append(input[seqitem])

        if var == "speed": action, idx, nxt = 'run', 4, 2
        if var == "wait": action, idx, nxt = 'stay', 3, 1
        if var == "run": action, idx, nxt = 'run', 3, 2

        
        for index, elem in enumerate(_seq):
            if elem != []:
                if all(i*300 < sublist[0] < (i+1)*300 for sublist in elem):
                    try:
                        #data.append([sublist[idx] if sublist[1] == action else np.nan for sublist in elem])
                        # print([sublist[:3] if sublist[1] == action else np.nan for sublist in elem], [sublist[idx] if sublist[1] == action else np.nan for sublist in _seq[index+2]][-1])
                        # print([sublist[0] for sublist in elem], i, [sublist[idx] if sublist[1] == action else np.nan for sublist in elem])

                        avgReward = np.nanmean([sublist[2] if sublist[1] == "run" else np.nan for sublist in elem])*100
                        nextitem = [abs([sublist[idx] if sublist[1] == action else np.nan for sublist in _seq[index+nxt]][-1])]
                        if not np.isnan(nextitem):
                            if avgReward in resblock.keys(): resblock[avgReward].extend(nextitem)
                            if avgReward not in resblock.keys(): resblock[avgReward] = nextitem
                    except IndexError: pass #pb with buffer items (-1, -1, -1, -1, -1)print(elem)

                elif any(i*300 < sublist[0] < (i+1)*300 for sublist in elem):# and not all(i*300 < sublist[0] < (i+1)*300 for sublist in elem):
                    if [sublist[0] for sublist in elem][0] > i * 300:
                        try:
                            avgReward = np.nanmean([sublist[2] if sublist[1] == "run" else np.nan for sublist in elem])*100
                            nextitem = [abs([sublist[idx] if sublist[1] == action else np.nan for sublist in _seq[index+nxt]][-1])]
                            if not np.isnan(nextitem):
                                if avgReward in resinter.keys(): resinter[avgReward].extend(nextitem)
                                if avgReward not in resinter.keys(): resinter[avgReward] = nextitem

                            # inter.append([sublist[idx] if sublist[1] == action else np.nan for sublist in elem])
                            # print([sublist[0] for sublist in elem], "TRANSITION", i)#, i*300, (i+1)*300)
                        except IndexError: pass #pb with buffer items (-1, -1, -1, -1, -1)
        
        cut = 0
        for key in resblock.keys():
            axs[2*i].scatter(key/100, np.nanmean(resblock[key]), marker='o', color='red', s=30, zorder=3)
            axs[2*i].scatter(key/100, np.nanmedian(resblock[key]), marker='o', color='cyan', s=30, zorder=3)
            axs[2*i].scatter([np.random.normal(key/100, 0.01, len(resblock[key]))], resblock[key], s=2.5, color='k' if len(resblock[key]) >= cut else 'grey');
            # medianblock[key], errorblock[key], dfblock[key] = np.nanmedian(resblock[key]), stats.sem(resblock[key]), len(resblock[key])
            # axs[2*i].errorbar(key/100, medianblock[key], yerr=errorblock[key], capsize=0, color='black', elinewidth=5, zorder=2)
            # axs[2*i].scatter(key/100, medianblock[key], marker='o', color='red', s=30, zorder=3)
            # axs[2*i].scatter(key/100, np.nanmean(resblock[key]), marker='o', color='green', s=30, zorder=3)
            # axs[2*i].scatter([np.random.normal(key/100, 0.05, len(resblock[key]))], resblock[key], s=5);
            # # axs[2*i].annotate(f'{medianblock[key]:.1f} ± {errorblock[key]:.1f}', ((key/100)-.25, medianblock[key]-(medianblock[key]/10)))

        for key in resinter.keys():
            axs[2*i+1].scatter(key/100, np.nanmean(resinter[key]), marker='o', color='red', s=30, zorder=3)
            axs[2*i+1].scatter(key/100, np.nanmedian(resinter[key]), marker='o', color='cyan', s=30, zorder=3)
            axs[2*i+1].scatter([np.random.normal(key/100, 0.01, len(resinter[key]))], resinter[key], s=2.5, color='k' if len(resinter[key]) >= cut else 'grey');
            # medianinter[key], errorinter[key], dfinter[key] = np.nanmedian(resinter[key]), stats.sem(resinter[key]), len(resinter[key])
            # axs[2*i+1].errorbar(key/100, medianinter[key], yerr=errorinter[key], capsize=0, color='black', elinewidth=5, zorder=2)
            # axs[2*i+1].scatter(key/100, medianinter[key], marker='o', color='red', s=30, zorder=3)
            # axs[2*i+1].scatter(key/100, np.nanmean(resinter[key]), marker='o', color='green', s=30, zorder=3)
            # axs[2*i+1].scatter([np.random.normal(key/100, 0.05, len(resinter[key]))], resinter[key], s=5);
            # axs[2*i+1].annotate(f'{medianinter[key]:.1f} ± {errorinter[key]:.1f}', ((key/100)+1.75, medianinter[key]-(medianinter[key]/10)))

        def _regression(block, ax, cut=15):
            xx = list(block.values())
            yy = list(block.keys())
            x, y = [], []
            for i in range(len(xx)):
                if len(xx[i]) >= cut:
                    x.extend(xx[i])
                    y.extend([yy[i] / 100] * len(xx[i]))
            gradient, intercept, r_value, p_value, std_err = stats.linregress(y, x)
            ax.plot(np.linspace(np.min(y), np.max(y), 500), 
                    gradient * np.linspace(np.min(y), np.max(y), 500) + intercept, 
                    'g' if p_value < 0.05 else 'r', 
                    label=f'mean_regr r: {r_value:.2f}, p: {p_value:.2f}, y = {gradient:.3f}x + {intercept:.2f}')
            ax.legend()
            return ax

        def _regression2(block, ax, cut=15, c='cyan'):
            xx = list(block.values())
            yy = list(block.keys())
            x, y = [], []
            for i in range(len(xx)):
                if len(xx[i]) >= cut:
                    x.extend(xx[i])
                    y.extend([yy[i] / 100] * len(xx[i]))
            a, b = stats.siegelslopes(x, y)
            ax.plot(np.linspace(np.min(y), np.max(y), 500), 
                    a * np.linspace(np.min(y), np.max(y), 500) + b, 
                    c, 
                    label=f'med_regr y = {a:.3f}x + {b:.2f}')
            ax.legend()
            return ax
            
        resinterfilter = {k: v for k, v in resinter.items() if 20 <= k <= 80}

        try:
            _regression(resblock, axs[2*i], cut=cut)
            _regression(resinter, axs[2*i+1], cut=cut)
            _regression2(resblock, axs[2*i], cut=cut)
            _regression2(resinter, axs[2*i+1], cut=cut)
            _regression2(resinterfilter, axs[2*i+1], cut=cut, c='g')


        except ValueError: pass # ValueError in firsttwo because we donc go to 3rd block


        alph = 90 if i%2==0 else 10
        axs[2*i].axvspan(-1, axes[1], color='gray', alpha=alph/250, zorder=1)
        axs[2*i].set_ylim(axes[2], axes[3])
        axs[2*i].set_xlim(axes[0], axes[1])

        axs[2*i+1].set_ylim(axes[2], axes[3])
        axs[2*i+1].set_xlim(axes[0], axes[1])



    axs[0].set_xlabel(f'Rwd over {targetlist} prev. runs')
    axs[0].set_ylabel(var)
    return axs


# pool all 90/-/10/- together
def plotseqinterpool(axs, input, targetlist, var, dat, axes=[0, 50, 0, 120]):

    _seq = np.empty([len(input), 0]).tolist()
    rewardProbaBlock = [90, 10, 90, 10, 90, 10, 90, 10, 90, 10, 90, 10]
    resblock10, resinter1090, resblock90, resinter9010 = dict(), dict(), dict(), dict()
    cls9010, cls1090 = dict(), dict()

    for elem in range(2*targetlist, len(input)):
        for seqitem in range(elem - 2*targetlist, elem):
            _seq[elem].append(input[seqitem])

    if var == "speed": action, idx, nxt = 'run', 4, 2
    if var == "wait": action, idx, nxt = 'stay', 3, 1
    if var == "run": action, idx, nxt = 'run', 3, 2

    for index, elem in enumerate(_seq):
        if elem != []:
            _list = [rewardProbaBlock[get_block(sublist[0])] if get_block(sublist[0]) is not None else None for sublist in elem]

            if all(_ == 90 for _ in _list):
                try:
                    avgReward = np.nanmean([sublist[2] if sublist[1] == "run" else np.nan for sublist in elem])*100
                    nextitem = [abs([sublist[idx] if sublist[1] == action and sublist[0] <= dat else np.nan for sublist in _seq[index+nxt]][-1])]
                    if not np.isnan(nextitem):
                        if avgReward in resblock90.keys(): resblock90[avgReward].extend(nextitem)
                        if avgReward not in resblock90.keys(): resblock90[avgReward] = nextitem
                except IndexError: pass

            elif all(_ == 10 for _ in _list):
                try:
                    avgReward = np.nanmean([sublist[2] if sublist[1] == "run" else np.nan for sublist in elem])*100
                    nextitem = [abs([sublist[idx] if sublist[1] == action and sublist[0] <= dat else np.nan for sublist in _seq[index+nxt]][-1])]
                    if not np.isnan(nextitem):
                        if avgReward in resblock10.keys(): resblock10[avgReward].extend(nextitem)
                        if avgReward not in resblock10.keys(): resblock10[avgReward] = nextitem
                except IndexError: pass

            else:
                if _list[0] == 90 and _list[-1] == 10:
                    try:
                        avgReward = np.nanmean([sublist[2] if sublist[1] == "run" else np.nan for sublist in elem])*100
                        nextitem = [abs([sublist[idx] if sublist[1] == action and sublist[0] <= dat else np.nan for sublist in _seq[index+nxt]][-1])]
                        blockcolor = [np.mean(_list) / 90]
                        if not np.isnan(nextitem): 
                            if avgReward in resinter9010.keys():
                                resinter9010[avgReward].extend(nextitem)
                                cls9010[avgReward].extend(blockcolor)
                            if avgReward not in resinter9010.keys():
                                resinter9010[avgReward] = nextitem
                                cls9010[avgReward] = blockcolor
                    except IndexError: pass

                elif _list[0] == 10 and _list[-1] == 90:
                    if not any([_ == '0.04' for _ in list(np.concatenate(elem).flat)]): 
                        try:
                            avgReward = np.nanmean([sublist[2] if sublist[1] == "run" else np.nan for sublist in elem])*100
                            nextitem = [abs([sublist[idx] if sublist[1] == action and sublist[0] <= dat else np.nan for sublist in _seq[index+nxt]][-1])]
                            blockcolor = [np.mean(_list) / 90]
                            if not np.isnan(nextitem):
                                if avgReward in resinter1090.keys():
                                    resinter1090[avgReward].extend(nextitem)
                                    cls1090[avgReward].extend(blockcolor)
                                if avgReward not in resinter1090.keys():
                                    resinter1090[avgReward] = nextitem
                                    cls1090[avgReward] = blockcolor
                        except IndexError: pass

    cut = 10 
    for key in resblock90.keys():
        axs[0].scatter(key/100, np.nanmean(resblock90[key]), marker='o', color='red', s=30, zorder=3)
        axs[0].scatter(key/100, np.nanmedian(resblock90[key]), marker='o', color='cyan', s=30, zorder=3)
        axs[0].scatter([np.random.normal(key/100, 0.01, len(resblock90[key]))], resblock90[key], s=2.5, color='k' if len(resblock90[key]) >= cut else 'grey');

    for key in resinter9010.keys():
        axs[1].scatter(key/100, np.nanmean(resinter9010[key]), marker='o', color='red', s=30, zorder=3)
        axs[1].scatter(key/100, np.nanmedian(resinter9010[key]), marker='o', color='cyan', s=30, zorder=3)
        axs[1].scatter([np.random.normal(key/100, 0.01, len(resinter9010[key]))], resinter9010[key], s=2.5, c= [(1-_, 0, _) for _ in cls9010[key]] if len(resinter9010[key]) >= cut else 'grey');

    for key in resblock10.keys():
        axs[2].scatter(key/100, np.nanmean(resblock10[key]), marker='o', color='red', s=30, zorder=3)
        axs[2].scatter(key/100, np.nanmedian(resblock10[key]), marker='o', color='cyan', s=30, zorder=3)
        axs[2].scatter([np.random.normal(key/100, 0.01, len(resblock10[key]))], resblock10[key], s=2.5, color='k' if len(resblock10[key]) >= cut else 'grey');

    for key in resinter1090.keys():
        axs[3].scatter(key/100, np.nanmean(resinter1090[key]), marker='o', color='red', s=30, zorder=3)
        axs[3].scatter(key/100, np.nanmedian(resinter1090[key]), marker='o', color='cyan', s=30, zorder=3)
        axs[3].scatter([np.random.normal(key/100, 0.01, len(resinter1090[key]))], resinter1090[key], s=2.5, c=[(1-_, 0, _) for _ in cls1090[key]] if len(resinter1090[key]) >= cut else 'grey');


    def _regression(block, ax, cut=15):
        xx = list(block.values())
        yy = list(block.keys())
        x, y = [], []
        for i in range(len(xx)):
            if len(xx[i]) >= cut:
                x.extend(xx[i])
                y.extend([yy[i] / 100] * len(xx[i]))
        gradient, intercept, r_value, p_value, std_err = stats.linregress(y, x)
        ax.plot(np.linspace(np.min(y), np.max(y), 500), 
                gradient * np.linspace(np.min(y), np.max(y), 500) + intercept, 
                'r' if p_value < 0.05 else 'r', 
                label=f'mean_regr r: {r_value:.2f}, p: {p_value:.2f}, y = {gradient:.3f}x + {intercept:.2f}')
        ax.legend()
        return ax

    def _regression2(block, ax, cut=15, c='cyan'):
        xx = list(block.values())
        yy = list(block.keys())
        x, y = [], []
        for i in range(len(xx)):
            if len(xx[i]) >= cut:
                x.extend(xx[i])
                y.extend([yy[i] / 100] * len(xx[i]))
        a, b = stats.siegelslopes(x, y)
        ax.plot(np.linspace(np.min(y), np.max(y), 500), 
                a * np.linspace(np.min(y), np.max(y), 500) + b, 
                c, 
                label=f'med_regr y = {a:.3f}x + {b:.2f}')
        ax.legend()
        return ax

    resinter9010filter = {k: v for k, v in resinter9010.items() if 20 <= k <= 80}
    resinter1090filter = {k: v for k, v in resinter1090.items() if 20 <= k <= 80}
        
    try:
        _regression(resblock90, axs[0], cut=cut)
        _regression(resinter9010, axs[1], cut=cut)
        _regression(resblock10, axs[2], cut=cut)
        _regression(resinter1090, axs[3], cut=cut)

        _regression2(resblock90, axs[0], cut=cut)
        _regression2(resinter9010, axs[1], cut=cut)
        _regression2(resblock10, axs[2], cut=cut)
        _regression2(resinter1090, axs[3], cut=cut)

        _regression2(resinter9010filter, axs[1], cut=cut, c='g')
        _regression2(resinter1090filter, axs[3], cut=cut, c='g')

    except ValueError: pass # ValueError in firsttwo because we donc go to 3rd block
    


    axs[0].axvspan(-1, axes[1], color='gray', alpha=90/250, zorder=1)
    axs[0].set_xlabel(f'Rwd over {targetlist} prev. runs')
    axs[0].set_ylabel(var)
    axs[0].set_ylim(axes[2], axes[3])
    axs[0].set_xlim(axes[0], axes[1])
    axs[0].set_title("blocks90")

    axs[1].set_ylim(axes[2], axes[3])
    axs[1].set_xlim(axes[0], axes[1])
    axs[1].set_title("90 -> 10")

    axs[2].axvspan(-1, axes[1], color='gray', alpha=10/250, zorder=1)
    axs[2].set_ylim(axes[2], axes[3])
    axs[2].set_xlim(axes[0], axes[1])
    axs[2].set_title("blocks10")

    axs[3].set_ylim(axes[2], axes[3])
    axs[3].set_xlim(axes[0], axes[1])
    axs[3].set_title("10 -> 90")

    return axs



#######################################################################
#######################################################################
#######################################################################
# WAITING TIME MODEL


# separate the data into time and reward bins
def prepare_data(sequence, animalList, sessionList, memsize=3, time_bins=6):
    """prepare data for fitting
    cut the data into time bins and reward bins"""
    bin_size = 3600/time_bins
    targetlist = generate_targetList(memsize)[::-1]
    temp_data = {}
    for time_bin in range(time_bins):
        temp_data[time_bin] = {}
        for animal in animalList:
            temp_data[time_bin][animal] = {k:[] for k in meankeys(targetlist)}
            for session in matchsession(animal, sessionList):
                temp_data[time_bin][animal] = combine_dict(temp_data[time_bin][animal], get_waiting_times(sequence[animal, session], memsize=memsize, filter=[time_bin*bin_size, (time_bin+1)*bin_size]))
    
    data = {}
    for animal in animalList:
        data[animal] = np.zeros((time_bins, len(meankeys(targetlist)))).tolist()
        for avg_bin, avg in enumerate(meankeys(targetlist)):  # 1 -> 0
            for time_bin in range(time_bins):
                data[animal][time_bin][avg_bin] = np.asarray(temp_data[time_bin][animal][avg])
    return data

# separate the data into time and reward bins for each session
def prepare_data_by_session(sequence, animalList, sessionList, memsize=3, time_bins=6):
    """prepare data for fitting
    cut the data into time bins and reward bins for each session"""
    bin_size = 3600/time_bins
    targetlist = generate_targetList(memsize)[::-1]
    temp_data = {}
    for bin in range(time_bins):
        temp_data[bin] = {}
        for animal in animalList:
            temp_data[bin][animal] = {key: {k:[] for k in meankeys(targetlist)} for key in matchsession(animal, sessionList)}
            for session in matchsession(animal, sessionList):
                temp_data[bin][animal][session] = combine_dict(temp_data[bin][animal][session], get_waiting_times(sequence[animal, session], memsize=memsize, filter=[bin*bin_size, (bin+1)*bin_size]))
    
    data = {}
    for animal in animalList:
        data[animal] = {}
        for session in matchsession(animal, sessionList):
            data[animal][session] = np.zeros((time_bins, len(meankeys(targetlist)))).tolist()
            for i, avg in enumerate(meankeys(targetlist)):  # 1 -> 0
                for bin in range(time_bins):
                    data[animal][session][bin][i] = np.asarray(temp_data[bin][animal][session][avg])
    return data

# plot session track without analysis files
def plot_animal_trajectory(root, animal, session, params, barplotaxes,
                xyLabels=["", ""], title=None, ax=None):
    ''' read position file and plot animal trajectory
    '''
    if ax is None: ax = plt.gca()
    time = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[3])[:90000]
    pos  = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[4])[:90000]/11
    mask = stitch([get_from_pickle(root, animal, session, name="mask.p")])[0]   

    running_Xs = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(pos, mask)]]
    idle_Xs = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(pos, mask)]]

    for i in range(0, len(params['blocks'])):
        ax.axvspan(params['blocks'][i][0], params['blocks'][i][1],
                    color='grey', alpha=params['rewardProbaBlock'][i]/250,
                    label="%reward: " + str(params['rewardProbaBlock'][i])
                    if (i == 0 or i == 1) else "")

    ax.plot(time, running_Xs, label="run", color="dodgerblue")
    ax.plot(time, idle_Xs, label="wait", color="orange")

    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    ax.set_xlim([barplotaxes[0], barplotaxes[1]])
    ax.set_ylim([barplotaxes[2], barplotaxes[3]])
    ax.set_xticks(np.arange(barplotaxes[0], barplotaxes[1]+1, 300))
    ax.set_xticklabels(np.arange(0, 61, 5))
    return ax


# plot variable median/mean fir each blockFdodger
def plot_median_per_bin(data, rewardProbaBlock, blocks, barplotaxes, color, stat,
                xyLabels=[" ", " ", " ", " "], title="", scatter=False, ax=False):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    if not ax:
        ax = plt.gca()
    for i in range(0, len(blocks)):
        ax.axvspan(blocks[i][0]/60, blocks[i][1]/60, color='grey', alpha=rewardProbaBlock[i]/250, label="%reward: " + str(rewardProbaBlock[i]) if (i == 0 or i == 1) else "")
        if scatter:
            ax.scatter(np.random.normal(((blocks[i][1] + blocks[i][0])/120), 1, len(data[i])), data[i], s=5, color=color[0])

    if stat == "Avg. ":
        ax.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.mean(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])
        if isinstance(data[0], list):
            ax.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.mean(data[i]) for i in range(0, len(blocks))], yerr=[stats.sem(data[i]) for i in range(0, len(blocks))], fmt='o', color=color[0], ecolor='black', elinewidth=1, capsize=0);

    elif stat == "Med. ":
        ax.plot([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.median(data[i]) for i in range(0, len(blocks))], marker='o', ms=7, linewidth=2, color=color[0])
        if isinstance(data[0], list):
            ax.errorbar([(blocks[i][1] + blocks[i][0])/120 for i in range(0, len(blocks))], [np.median(data[i]) for i in range(0, len(blocks))], yerr=[stats.sem(data[i]) for i in range(0, len(blocks))], fmt='o', color=color[0], ecolor='black', elinewidth=1, capsize=3);

    ax.set_title(title)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(stat + xyLabels[1])
    ax.set_xlim([barplotaxes[0], barplotaxes[1]])
    ax.set_ylim([barplotaxes[2], barplotaxes[3]])
    return ax
    


# raster of (non)rewarded trials, reward average selection, and idle time distribution plots
def plot_rewards(data, avg, memsize=3, ax=None, filter=[0, 3600]):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    input = bin_seq(data)

    # get the number of runs per block to init vars
    c = np.zeros(12)
    for i in range(12):
        n = 0
        for a in range(len(input[i])):
            if input[i][a][1] == "run": n += 1
        c[i] = n

    # get the 0-1 rewards and times for each block
    rewards = np.ones((12, int(max(c))))*.5
    times = np.ones((12, int(max(c))))*.5
    last_rewards_from_previous_block = np.ones((12, 2))*.5

    for i in range(12):
        rw, trw = [], []
        for a in range(0, len(input[i])):
            if input[i][a][1] == "run":
                rw.append(input[i][a][2])
                trw.append(input[i][a][0])
        # cosmetic, get the last two rewards of the previous block before the first rewards of the next block
        if i < 11:
            last_rewards_from_previous_block[i+1] = rw[-2:]

        rewards[i, 0:len(rw)] = rw
        times[i, 0:len(trw)] = trw


    cmap = ListedColormap(['r', 'w', 'g'])
    edges = np.copy(rewards,)
    edges = np.where(edges == 0.0, 'k', edges)
    edges = np.where(edges == '1.0', 'k', edges)
    edges = np.where(edges == '0.5', 'w', edges)
    # edges = [item for sublist in edges for item in sublist]


    markers = np.copy(rewards,)
    markers = np.where(markers == 0.0, '$x$', markers)
    markers = np.where(markers == '1.0', '$✓$', markers)
    markers = np.where(markers == '0.5', '', markers)
    # markers = [item for sublist in markers for item in sublist]
    unique_markers = np.unique(markers)


    x = np.arange(max(c))
    y = np.arange(12)[::-1]
    X, Y = np.meshgrid(x, y)

    for um in unique_markers:
        mask = np.array(markers) == um
        ax.scatter(X[mask], Y[mask], s=200, marker=um, c=rewards[mask], cmap=cmap, vmin=0, vmax=1, edgecolors=edges[mask], linewidths=1)

    xlast = [-2, -1]
    ylast = np.arange(12)[::-1]
    Xlast, Ylast = np.meshgrid(xlast, ylast)

    lastedges = np.copy(last_rewards_from_previous_block,)
    lastedges = np.where(lastedges == 0.0, 'k', lastedges)
    lastedges = np.where(lastedges == '1.0', 'k', lastedges)
    lastedges = np.where(lastedges == '0.5', 'w', lastedges)

    lastmarkers = np.copy(last_rewards_from_previous_block,)
    lastmarkers = np.where(lastmarkers == 0.0, '$x$', lastmarkers)
    lastmarkers = np.where(lastmarkers == '1.0', '$✓$', lastmarkers)
    lastmarkers = np.where(lastmarkers == '0.5', '', lastmarkers)



    for um in unique_markers:
        mask = np.array(lastmarkers) == um
        ax.scatter(Xlast[mask], Ylast[mask], s=200, marker=um, 
        c=last_rewards_from_previous_block[mask], 
        cmap=cmap, vmin=0, vmax=1, 
        edgecolors=lastedges[mask], 
        linewidths=1, alpha=0.35)
    # ax.scatter(Xlast, Ylast, s=200, marker='x', c=last_rewards_from_previous_block, cmap=cmap, vmin=0, vmax=1, edgecolors=edges, linewidths=1, alpha=0.35)

    # # plot a line between time blocks
    # for i in range(1, 11, 2):
    #     ax.axhline(i+0.5, xmin=0.045, color='k', linewidth=1)

    

    ax.set_xticks([])
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(np.arange(1, 13)[::-1])
    ax.spines['bottom'].set_color("none")
    ax.spines['left'].set_color("none")
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    ax.set_ylabel('# Block')
    ax.set_xlabel('# Reward')
    ax.set_title(f'Average reward: {avg}')
    ax.set_xlim(-5, int(max(c))+1)
    ax.set_title(f'Reward sequence in example session')


    def _get_waiting_times_idx(data, memsize=3):
        """get waiting times idx from data"""
        waiting_times = {k:[] for k in meankeys(generate_targetList(seq_len=memsize)[::-1])}
        idx=0
        for i in range(len(data)):
            if data[i][1] == 'stay':
                try:
                    avg_rwd = round(np.mean([data[i-n][2] for n in range(1, (memsize*2)+1, 2)]), 2)
                    waiting_times[avg_rwd].append(idx)
                except:  # put the first n waits in rwd=1 (because we don't have the previous n runs to compute the average reward)
                    waiting_times[1].append(idx)
                idx+=1
        return waiting_times

    # find the target avg
    timeres = []
    dtimeres = []
    res = _get_waiting_times_idx(data, memsize=memsize)[avg]
    

    # convert sequence index to 2D array index (block, run) (bc. we don't have same number of runs per block)
    cc = np.cumsum(c)
    cc = np.insert(cc, 0, 0)

    def _convert_res(res):
        idx, idy = 0, 0
        if cc[0] <= res < cc[1]: idx, idy = 0, int(res-cc[0])
        if cc[1] <= res < cc[2]: idx, idy = 1, int(res-cc[1])
        if cc[2] <= res < cc[3]: idx, idy = 2, int(res-cc[2])
        if cc[3] <= res < cc[4]: idx, idy = 3, int(res-cc[3])
        if cc[4] <= res < cc[5]: idx, idy = 4, int(res-cc[4])
        if cc[5] <= res < cc[6]: idx, idy = 5, int(res-cc[5])
        if cc[6] <= res < cc[7]: idx, idy = 6, int(res-cc[6])
        if cc[7] <= res < cc[8]: idx, idy = 7, int(res-cc[7])
        if cc[8] <= res < cc[9]: idx, idy = 8, int(res-cc[8])
        if cc[9] <= res < cc[10]: idx, idy = 9, int(res-cc[9])
        if cc[10] <= res < cc[11]: idx, idy = 10, int(res-cc[10])
        if cc[11] <= res < cc[12]: idx, idy = 11, int(res-cc[11])
        return idx, idy

    for r in res:
        idx, idy = _convert_res(r)
        didx, didy = _convert_res(r-memsize+1)

        if filter[0] <= times[idx, idy] <= filter[1]:
            timeres.append(times[idx, idy])  ## 2D array index for time of the end of the sequence in the data
            dtimeres.append(times[didx, didy])  ## 2D array index for time of the start of the sequence in the data
            ax.add_patch(patches.FancyBboxPatch((idy-(memsize-1)-0.1, 11-idx), memsize-.8, .04, boxstyle=patches.BoxStyle("Round", pad=.35), fill=False, lw=2.5, color='k'))
        else:
            ax.add_patch(patches.FancyBboxPatch((idy-(memsize-1)-0.1, 11-idx), memsize-.8, .04, boxstyle=patches.BoxStyle("Round", pad=.35), fill=False, lw=2.5, color='k', alpha=0.35))

    nextwait = []
    sequenceduration = []
    for t, dt in zip(timeres, dtimeres):
        start, end = 0, 0
        for elem in data:
            if data[elem][0] == t:
                if elem+1 < len(data):
                    if data[elem+1][3] != 0:
                        nextwait.append(data[elem+1][3])  # 1st wait time after the sequence
                    end = data[elem+1][0]
                break
            if data[elem][0] == dt:
                if elem+1 < len(data):
                    start = data[elem][0]
        sequenceduration.append(end-start)

    return nextwait




def plot_rewards_distribution(nextwait, avg, color, memsize=3, ax=None, label=''):
    if ax is not None:
        mx = 300
        bins = np.arange(0, mx+1, .5)


        ax[0].hist(sorted(nextwait)[::-1], bins=bins, histtype='step', color=color, lw=2, 
                    density=True, 
                    weights=np.ones(len(nextwait)) / len(nextwait) *100,
                    label=label)
        ax[0].set_title(f"Idle time distribution after {avg}\nrewards obtained in 0-10 min")
        ax[0].set_xlabel("Idle time (s)")
        ax[0].set_ylabel("PDF")
        ax[0].set_xlim(0, 25)
        ax[0].set_ylim(0, 1.1)


        ax[1].hist(sorted(nextwait)[::-1], bins=bins, histtype='step', color=color, lw=2,
                    density=True, 
                    weights=np.ones(len(nextwait)) / len(nextwait) *100,
                    cumulative=-1, 
                    label=label)
        ax[1].set_title(f"Log-log Idle time distribution after {avg}\nrewards obtained in 0-10 min")
        ax[1].set_xlabel("log(Idle time) (s)")
        ax[1].set_ylabel("log(1-CDF)")
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_xlim(0.1, 1000)
        ax[1].set_ylim(0.001, 1.1)

        ax[0].legend()
        ax[1].legend()



def generate_targetList(seq_len=1):
    """generate list of all reward combinations for specified memory length
    call: generate_targetList(seq_len=2)"""
    get_binary = lambda x: format(x, 'b')
    output = []
    for i in range(2**seq_len):
        # list binary number from 0 to 2**n, add leading zeroes when resulting seq is too short 
        binstr = "0" * abs(len(get_binary(i)) - seq_len) + str(get_binary(i))
        output.append(binstr)
    return output

def meankeys(targetlist):
    """get each possible mean for a list of targets
    call: meankeys(generate_targetList(seq_len=2))"""
    result = []
    for target in targetlist:
        res = round(np.mean([int(elem) for elem in target]), 2)
        if res not in result:
            result.append(res)
    return result

def get_waiting_times(data, memsize=3, filter=[0, 3600], toolong=3600):
    """get waiting times from sequence of actions data and separate them
    according to the average reward of the sequence"""
    waiting_times = {k:[] for k in meankeys(generate_targetList(seq_len=memsize)[::-1])}
    for i in range(len(data)):
        if data[i][1] == 'stay':
            if filter[0] <= data[i][0] <= filter[1] and data[i][3] != 0:
                if data[i][3] < toolong:  # filter out
                    try:
                        avg_rwd = round(np.mean([data[i-n][2] for n in range(1, (memsize*2)+1, 2)]),2)
                        waiting_times[avg_rwd].append(data[i][3])
                    except:  # put the first n waits in rwd=1 (because we don't have the previous n runs to compute the average reward)
                        waiting_times[1].append(data[i][3])
    return waiting_times

def combine_dict(d1, d2):
    """combine two dictionaries with the same keys"""
    keys = d1.keys()
    values = [np.concatenate([d1[k], d2[k]]) for k in keys]
    return dict(zip(keys, values))


# def log_tick_formatter(val, pos=None):
#     '''Return the string representation of 10^val'''
#     return r'$10^{%s}$' % val

# def plot_polygon(x, y, z, ax, color='k', limitZ=-3, limitX=-1):
#     '''plot the distribution of the data as a polygon'''
#     z[-1] = limitZ  # force last point to be at 10^-3 instead of -inf
#     x[0] = limitX  # force first point to be at 10^-1 instead of 0
#     for i in range(len(x)-1):
#         ax.plot([x[i], x[i+1]], [y, y], [z[i], z[i]], color=color)
#         ax.plot([x[i+1], x[i+1]], [y, y], [z[i], z[i+1]], color=color)

def plot_full_distribution(data, animal, plot_fit=False, N_bins=6, N_avg=4):
    '''plot the full distribution of the data'''
    ###
    # NOT SAME NUMBER OF OBSERVATIONS IN EACH CURVE, BUT SAME NORMALIZATION ???
    ###

    def _plot_wald_fitted(waits, p, ax=None, color='k', plot_fit=True, label='', lw=2):
        """plot fitted wald distribution without fitting"""
        if ax is None: ax = plt.gca()
        waits = np.asarray(waits)

        bins=np.linspace(0, waits.max(), int(max(waits)))
        ydata, xdata, _ = ax.hist(waits, bins=bins,
                        color=color, alpha=1, zorder=1, 
                        density=True, # weights=np.ones_like(waits) / len(waits),
                        histtype="step", lw=lw, cumulative=-1, label=label)

        if plot_fit:
            x = np.linspace(0.001, 500, 10000)
            ax.plot(x, 1-Wald_cdf(x, *p), color=color, lw=2, zorder=4, ls='--', label=f'{label} fit')
        return ax

    fig, axs = plt.subplots(1, N_bins, figsize=(3*N_bins, 3))
    (alpha, theta, gamma, alpha_t, thetaprime, gamma_t, alpha_R, thetasecond, gamma_R), loss = modelwald_fit(data[animal])

    lbls = ['1', '0.67', '0.33', '0']
    for j in range(N_bins):
        for i in range(N_avg):
            color = plt.get_cmap('inferno')(i / N_avg)
            lw = 3.5 if j == 0 and i == 1 else 2
            _plot_wald_fitted(data[animal][j][i], 
                            (alpha + j*alpha_t + i*alpha_R, theta, gamma + j*gamma_t + i*gamma_R), 
                            ax=axs[j], color=color, plot_fit=plot_fit, label=lbls[i], lw=lw)
        axs[j].set_xlim(.1, 1000)
        axs[j].set_ylim(.001, 1.1)
        axs[j].set_xscale("log")
        axs[j].set_yscale("log")
        axs[j].set_xlabel('log(idle time) (s)')
        axs[j].set_ylabel('log(1-CDF)')
        axs[j].set_title(f'{j*10}-{(j+1)*10} min')
        axs[j].legend()

######################################################

def plot_DDMexample(mean, std, A, t0, N=100, title=None):
    """plot example of DDM with specified parameters"""

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    trials = [generate_trials(mean, std, A, 0) for _ in range(N)]
    
    example_plot = True
    for dv in trials: 
        dv[-1] = A
        x = np.arange(len(dv))/25
        y = dv
        if len(y) > 5*25 and example_plot:
            ax.plot(x, y, c='k', lw=1.5, zorder=4)
            ax.annotate(r'$t_f$', (len(y)/25, A-2), (0, 1), xycoords="data", textcoords="offset points", color="k", zorder=4, fontsize=14)
            example_plot = False
        ax.plot(x, y, c='orange', alpha=.5, zorder=3)
    
    waits = np.array([len(t)/25 for t in trials], dtype=np.float64)

    waitmean = A / mean * np.tanh(mean * A)  #  + t0
    ax.plot(np.linspace(0, waitmean/25, int(waitmean)+1), A / waitmean * np.arange(waitmean), c="r", zorder=4)
    ax.annotate(r'$v$', ((waitmean/25)/2-1, (A/2)+1), (0, 1), xycoords="data", textcoords="offset points", color="r", zorder=4, fontsize=14)
    # ax.spines['left'].set_position(('data', t0))
    # ax.axhline(0, xmin=t0, c="k", ls="--", zorder=5, lw=2.5)
    ax.axhline(A, c='c', zorder=5, lw=2.5)
    ax.set_yticks([0, A])
    ax.set_yticklabels([0, r'$A$'])
    ax.get_yticklabels()[1].set_color('c') 
    ax.set_xlabel('t')
    ax.set_ylabel('dv')
    ax.set_title(title)
    ax.set_ylim(-10, 30)
    ax.set_xlim(-2, 25)
    ax.plot((0, -t0), (0, 0), c="g", zorder=5, lw=2.5)
    ax.annotate(r'$t_0$', ((0-t0), 1), (0, 0), xycoords="data", textcoords="offset points", color="g", zorder=4, fontsize=14)


    # inset distribution
    l, b, h, w = 0.105, .55, .5, .85
    ax1 = fig.add_axes([l, b, w, h])
    mx = 300
    bins = np.arange(0, mx+1, .5)
    ax1.hist(waits, bins=bins, color='k',
                    alpha=.5, zorder=4, histtype="step", lw=2,
                    # cumulative=1, 
                    density=True,
                    weights=np.ones_like(waits) / len(waits),
                    )
                    
    p, _ = wald_fit(waits)
    x = np.linspace(0, 1000, 10000)
    ax1.plot(x, Wald_pdf(x, *p), 'm-', label='Default')
    ax1.set_ylim(0, 0.8)
    ax1.set_xlim(-2, 25)
    ax1.set_ylabel('PDF')
    ax1.axis('off')

    # inset
    l, b, h, w = .7, .7, .25, .25
    ax2 = fig.add_axes([l, b, w, h])
    ax2.hist(waits, bins=bins, color='k',
                    alpha=.5, zorder=4, histtype="step", lw=2,
                    cumulative=-1, density=True,
                    weights=np.ones_like(waits) / len(waits),
                    )
    ax2.plot(x, 1-Wald_cdf(x, *p), 'm-', label='Default')
    ax2.set_ylim(0.001, 1.1)
    ax2.set_xlim(.1, 1000)
    ax2.set_ylabel('log 1-CDF')
    ax2.set_xlabel('log t')
    ax2.set_yscale('log')
    ax2.set_xscale('log')


def plot_DDMexampleParams(v, A):
    mean = v
    ax = plt.gca()
    N=250
    t0=2
    std=1
    # np.random.seed(0)
    trials = [generate_trials(mean, std, A, t0) for _ in range(N)]

    rnd = np.random.randint(0, len(trials))
    for idx, dv in enumerate(trials): 
        dv[-1] = A
        if idx == rnd:
            ax.plot(dv, c='k', lw=2, zorder=5)
        ax.plot(dv, c='orange', alpha=.5, zorder=3)
        

    waits = np.array([len(t) for t in trials], dtype=np.float64)

    bins = np.linspace(0, waits.max(), int(max(waits)))
    ax.hist(waits, bins=bins, color='k', bottom=A,
                    alpha=.5, zorder=4, histtype="step", lw=2,
                    weights=np.ones_like(waits) / len(waits)*25,
                    )

    p, _ = wald_fit(waits)
    x = np.linspace(0, 1000, 10000)
    ax.plot(x, (25*Wald_pdf(x, *p))+A, 'm-', label='Default')

    waitmean = A / mean * np.tanh(mean * A) + t0
    ax.plot(np.linspace(t0, waitmean, int(waitmean)+1), A / waitmean * np.arange(waitmean), c="r", zorder=4)
    ax.annotate(r'$v$', ((t0+waitmean)/2-1, (A/2)+1), (0, 1), xycoords="data", textcoords="offset points", color="r", zorder=4)
    ax.axhline(0, c="k", ls="--", zorder=4)
    ax.axhline(A, c='c', zorder=4)
    ax.set_yticks([0, A])
    ax.set_yticklabels([0, r'$A$'])
    ax.get_yticklabels()[1].set_color('c') 
    ax.set_xlabel('t')
    ax.set_ylabel('dv')
    ax.set_title('')
    ax.set_ylim(-10, 10)
    ax.set_xlim(0, 25)
    ax.plot((0, t0), (0, 0), c="g", zorder=4)
    ax.annotate(r'$t_0$', ((0+t0)/2, 1), (0, 1), xycoords="data", textcoords="offset points", color="g", zorder=4)


def generate_trials(mean, std, A, t0):
    """generate a single diffusion trial"""
    # np.random.seed(0)
    dv = [0] * (t0 + 1)
    while dv[-1] < A:
        evidence = np.random.normal(mean, std)
        dv.append(dv[-1] + evidence)
    return dv

#############################################################

def Wald_pdf(x, alpha, theta, gamma):
    """Wald pdf"""
    x = np.asarray(x) - theta  # x = x - theta
    x[x<0] = 1e-10
    arg = 2 * np.pi * x ** 3
    res = alpha / np.sqrt(arg) * np.exp(-((alpha-gamma * x) ** 2) / (2 * x))
    return np.array(res, dtype=np.float64)

def Wald_cdf(x, alpha, theta, gamma):
    """Wald cdf"""
    # from https://github.com/mark-hurlstone/RT-Distrib-Fit 
    x = x - theta
    x[x<0] = 1e-10
    return np.array(stats.norm.cdf((gamma*x-alpha)/np.sqrt(x)) + np.exp(2*alpha*gamma)*stats.norm.cdf(-(gamma*x+alpha)/np.sqrt(x)), dtype=np.float64)

# interactive plot
def plot_interactiveWald(alpha=1, gamma=2, t_0=0):
    """interactive plot of Wald pdf"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    x = np.linspace(0.01, 4, 400)
    axs[0].plot(x, Wald_pdf(x, 1, 0, 2), 'k-', label='Default')
    axs[0].plot(x, Wald_pdf(x, 2.5, 0, 2), 'c', label='increased alpha')
    axs[0].plot(x, Wald_pdf(x, 1, 0, 3.8), 'r-', label='increased gamma')
    axs[0].plot(np.linspace(0.81, 4, 1000), Wald_pdf(np.linspace(0.81, 4, 1000), 1, .8, 2), 'g-', label='increased theta')
    axs[0].set_ylabel('PDF')
    axs[0].set_xlabel('t')
    axs[0].set_xlim(0, 4)
    axs[0].set_ylim(0, 4)
    axs[0].legend()

    pdf = Wald_pdf(x, alpha, t_0, gamma)
    cdf = 1-Wald_cdf(x, alpha, t_0, gamma)
    axs[1].plot(x, pdf)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('pdf')
    axs[1].set_title('pdf')
    axs[2].plot(x, cdf)
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('log t')
    axs[2].set_ylabel('log 1-cdf')
    axs[2].set_title('log 1-cdf')

    axs[1].set_xlim(0, 4)
    axs[1].set_ylim(0, 4)
    axs[2].set_xlim(0.01, 10)
    axs[2].set_ylim(0.01, 1.1)
    return


##########################################################
def log_lik_wald(x, params, robustness_param=1e-20):
    """log likelihood function for Wald distribution"""
    alpha, theta, gamma = params
    pdf_vals = Wald_pdf(x, alpha, theta, gamma) + robustness_param
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit(params, *args, robustness_param=1e-20):
    """negative log likelihood function for Wald distribution"""
    alpha, theta, gamma = params
    x = args
    pdf_vals = Wald_pdf(x, alpha, theta, gamma) + robustness_param
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

def wald_fit(x, alpha_init=2, theta_init=0, gamma_init=.5):
    """fit Wald distribution"""
    params_init = np.array([alpha_init, theta_init, gamma_init])
    res = scipy.optimize.minimize(crit, params_init, args=x, bounds=((0, None), (0, 1e-8), (0, None)))
    return res.x, res.fun
    
def genWaldSamples(N, alpha, gamma, maximum=500):
    """generate Wald samples"""
    # 230x faster than drawfromDDM (pyDDM)
    # based on https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    x = np.linspace(1e-8, maximum, maximum*100)

    def p(x, alpha, gamma): 
        return alpha / np.sqrt(2 * np.pi * x ** 3) * np.exp(-((alpha-gamma * x) ** 2) / (2 * x))

    def normalization(x, alpha, gamma): 
        return scipy.integrate.simps(p(x, alpha, gamma), x)

    pdf = p(x, alpha, gamma)/normalization(x, alpha, gamma)
    cdf = np.cumsum(pdf); cdf /= max(cdf)

    u = np.random.uniform(0, 1, int(N))
    interp_function = scipy.interpolate.interp1d(cdf, x)
    samples = interp_function(u)
    return samples

def example_wald_fit(mean, std, A, t0, N=100, ax=None, color='k'):
    """example of fitting Wald distribution"""
    if ax is None: ax = plt.gca()
    waits=genWaldSamples(N, A, mean)
    bins=np.linspace(0, waits.max(), int(max(waits)))
    ydata, xdata, _ = ax.hist(waits, bins=bins,
                    color=color, alpha=.5, zorder=1, 
                    density=True, # weights=np.ones_like(waits) / len(waits),
                    histtype="step", lw=2, cumulative=-1, label=f'N={N} simulated samples')

    x = np.linspace(0.01, 500, 10000)
    xdata = xdata[:-1]

    fittime = time.time()
    (alpha, theta, gamma), lossWald = wald_fit(waits)
    ax.plot(x, 1-Wald_cdf(x, alpha, theta, gamma), color=color, lw=2, zorder=4, label=f'best fit')
    ydatapdf, xdatapdf, _ = ax.hist(waits, bins=bins, alpha=.0, zorder=1, density=True, histtype="step",)

    ax.set_xlim(1, 500)
    ax.set_ylim(.001, 1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('log Wait time')
    ax.set_ylabel('log 1-CDF')
    if mean == 0.1: ax.legend()

    return alpha, theta, gamma, lossWald

def plot_color_line(ax, x, y, z, cmap = 'viridis', vmin = None, vmax = None, alpha = 1, linewidth = 1, linestyle = '-', zorder = 1):
    """plot line with color based on z values"""
    from matplotlib.collections import LineCollection
    color = np.abs(np.array(z, dtype=np.float64))
    if vmin == None:
        vmin = np.nanmin(color)
    if vmax == None:
        vmax = np.nanmax(color)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha = alpha, linewidth = linewidth, linestyle = linestyle, zorder = zorder)
    lc.set_array(color)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)
    return line



######################################################################
# alpha, alpha', alpha'', gamma, gamma', gamma''
def model_crit(params, *args, robustness_param=1e-20):
    """negative log likelihood function for full model"""
    alpha, theta, gamma, alpha_t, theta_prime, gamma_t, alpha_R, theta_second, gamma_R = params
    neg_log_lik_val = 0
    N_bins, N_avg = args[1]
    ALPHA = np.zeros((N_bins, N_avg))
    GAMMA = np.zeros((N_bins, N_avg))
    _theta = theta + theta_prime + theta_second

    for bin in range(N_bins):
        for avg in range(N_avg):
            ALPHA[bin, avg] = alpha + bin*alpha_t + avg*alpha_R
            GAMMA[bin, avg] = gamma + bin*gamma_t + avg*gamma_R

    for bin in range(N_bins):
        for avg in range(N_avg):
            _alpha = ALPHA[bin, avg] if ALPHA[bin, avg] > 0 else 1e-8
            _gamma = GAMMA[bin, avg]# if GAMMA[bin, avg] > 0 else 1e-8
            try:
                pdf_vals = Wald_pdf(args[0][bin][avg], _alpha, _theta, _gamma)
                ln_pdf_vals = np.log(pdf_vals + robustness_param)
                log_lik_val = ln_pdf_vals.sum()

                n = len(args[0][bin][avg]) if len(args[0][bin][avg]) > 0 else 1
                neg_log_lik_val += (-log_lik_val / n)
            except:
                neg_log_lik_val += 0  # add 0 instead of throwing an error when there is no data in a bin*avg
    return neg_log_lik_val

# alpha, alpha', alpha'', gamma, gamma', gamma''
def model_compare(params, *args, robustness_param=1e-20):
    """BIC to compare models with different number of parameters and curves"""
    alpha, theta, gamma, alpha_t, theta_prime, gamma_t, alpha_R, theta_second, gamma_R = params
    BIC = 0
    N_bins, N_avg = args[1]
    N_params = args[2]
    ALPHA = np.zeros((N_bins, N_avg))
    GAMMA = np.zeros((N_bins, N_avg))
    _theta = theta + theta_prime + theta_second

    for bin in range(N_bins):
        for avg in range(N_avg):
            ALPHA[bin, avg] = alpha + bin*alpha_t + avg*alpha_R
            GAMMA[bin, avg] = gamma + bin*gamma_t + avg*gamma_R

    for bin in range(N_bins):
        for avg in range(N_avg):
            _alpha = ALPHA[bin, avg] if ALPHA[bin, avg] > 0 else 1e-8
            _gamma = GAMMA[bin, avg]# if GAMMA[bin, avg] > 0 else 1e-8
            try:
                pdf_vals = Wald_pdf(args[0][bin][avg], _alpha, _theta, _gamma)
                ln_pdf_vals = np.log(pdf_vals + robustness_param)
                log_lik_val = ln_pdf_vals.sum()

                n = len(args[0][bin][avg]) if len(args[0][bin][avg]) > 0 else 1
                k = N_params
                BIC += k * np.log(n) - 2 * log_lik_val
            except:
                BIC += 0  # add 0 instead of throwing an error when there is no data in a bin*avg
    return BIC

#params = a, t, g, a', t', g', a'', t'', g''
def modelwald_fit(data, init=[2, 0, .5, 0, 0, 0, 0, 0, 0], 
        f=model_crit, N_bins=6, N_avg=4, N_params=2,
        alpha_t_fixed=False, gamma_t_fixed=False, 
        alpha_R_fixed=False, gamma_R_fixed=False,
        ):
    """fit full model to data"""
    params_init = np.array(init)
    alpha_t_bounds = (None, None) if not alpha_t_fixed else (0, 1e-8)
    gamma_t_bounds = (None, None) if not gamma_t_fixed else (0, 1e-8)
    alpha_R_bounds = (None, None) if not alpha_R_fixed else (0, 1e-8)
    gamma_R_bounds = (None, None) if not gamma_R_fixed else (0, 1e-8)

    res = scipy.optimize.minimize(f, params_init, args=(data, [N_bins, N_avg], N_params), 
                                        bounds=((0, None), (0, 1e-8), (0, None), 
                                            alpha_t_bounds, (0, 1e-8), gamma_t_bounds, 
                                            alpha_R_bounds, (0, 1e-8), gamma_R_bounds))
    return res.x, res.fun




################################################

def plot_parameter_evolution(p, axs=None, N_bins=6, N_avg=4):

    (alpha, gamma, alpha_t, gamma_t, alpha_R, gamma_R) = p
    ALPHA = np.zeros((N_bins, N_avg))
    GAMMA = np.zeros((N_bins, N_avg))

    for bin in range(N_bins):
        for avg in range(N_avg):
            ALPHA[bin, avg] = alpha + bin*alpha_t + avg*alpha_R
            GAMMA[bin, avg] = gamma + bin*gamma_t + avg*gamma_R

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})

    X, Y = np.meshgrid(np.arange(N_avg), np.arange(N_bins))
    axs[0].plot_surface(X, Y, ALPHA, cmap='winter', edgecolor='none')
    axs[0].set_title(r'Value of $\mathrm{A}$')
    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(['1', '0.67', '0.33', '0'])
    axs[0].set_xlabel('Reward history', labelpad=5)
    axs[0].set_ylim([-0.5, 5.5])
    axs[0].set_yticks([0, 1, 2, 3, 4, 5])
    axs[0].set_yticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'], va='center', ha='left', rotation=-15)
    axs[0].set_ylabel('Time bin', labelpad=15)
    axs[0].set_zlabel(r'$\alpha$', labelpad=5)
    axs[0].set_zlim([.8, 2.2])
    axs[0].set_zticks([1, 1.5, 2])
    axs[0].set_zticklabels(['1.0', '1.5', '2.0'])
    axs[0].text(0., 5, 2., r"$\alpha R$: Effect of reward on $\mathrm{A}$", color='black', fontsize=12, zdir='x', zorder=10)
    axs[0].text(3, 0.5, 1.2, r"$\alpha t$: Effect of time on $\mathrm{A}$", color='black', fontsize=12, zdir=(0, 6, 1), zorder=10)
    axs[0].text(0, 0, 0.6, r"$\alpha_0$: Baseline $\mathrm{A}$", color='black', fontsize=12, zdir='x', zorder=10)

    axs[1].plot_surface(X, Y, GAMMA, cmap='autumn', edgecolor='none')
    axs[1].set_title(r'Value of $\Gamma$')
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(['1', '0.67', '0.33', '0'])
    axs[1].set_xlabel('Reward history')
    axs[1].set_ylim([-0.5, 5.5])
    axs[1].set_yticks([0, 1, 2, 3, 4, 5])
    axs[1].set_yticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'], va='center', ha='left', rotation=-15)
    axs[1].set_ylabel('Time bin')
    axs[1].set_zlabel(r'$\gamma$')
    axs[1].set_zlim([.0, .6])
    axs[1].set_zticks([.2, .4, .6, 0.8])
    axs[1].set_zticklabels(['0.2', '0.4', '0.6', '0.8'])
    axs[1].text(0, 5, 0, r"$\gamma R$: Effect of reward on $\Gamma$", color='black', fontsize=12, zdir=(4, 0, -.5), zorder=10)
    axs[1].text(0, -1, .6, r"$\gamma t$: Effect of time on $\Gamma$", color='black', fontsize=12, zdir=(0, -10, .1), zorder=10)
    axs[1].text(0, 0, 0.4, r"$\gamma_0$: Baseline $\Gamma$", color='black', fontsize=12, zdir='x', zorder=10)


################################################
def test_all_conds_between_themselves(conds, vars, ax=None):
    """dirty stats to test all conditions against each other"""
    if ax is None: ax = plt.gca()
    for idx, var in enumerate(vars):
        c = 0
        for i, cond1 in enumerate(conds):
            for j, cond2 in enumerate(conds):
                if i >= j:
                    continue
                data1 = [var[animal][cond1] for animal in list(var.keys())]
                data2 = [var[animal][cond2] for animal in list(var.keys())]
                s, p = stats.ttest_ind(data1, data2)
                # print(f"{idx} {cond1} vs {cond2}: {p:.3f} {'*' if p < .05 else ''}")

                if p < .05:
                    print(f"{idx} {cond1} vs {cond2}: {p:.3f} {'*' if p < .05 else ''}")
                    y = np.max([np.mean(data1)+ 2*np.std(data1), np.mean(data2)+ 2*np.std(data2)]) + c
                    ax[idx].plot((i, j), (y, y), color='k')
                    ax[idx].scatter((i+j)/2, y+.1, color='k', marker=r'$\ast$')
                    c += 0.1


def test_all_keys_between_themselves(losses, keys, ax=None):
    """dirty stats to test all conditions against each other, but with keys"""
    if ax is None: ax = plt.gca()
    c = 0
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i >= j:
                continue
            data1 = [losses[animal][key1]/losses[animal][False, False, False, False] for animal in list(losses.keys())]
            data2 = [losses[animal][key2]/losses[animal][False, False, False, False] for animal in list(losses.keys())]
            s, p = stats.ttest_ind(data1, data2)
            print(f"{key1} vs {key2}: {p:.3f} {'*' if p < .05 else ''}")

            if p < .05:
                y = np.max([np.mean(data1), np.mean(data2)]) + c
                ax.plot((i+2, j+2), (y, y), color='g')
                ax.scatter((i+j+4)/2, y, color='g', marker=r'$\ast$')
                c += 0.001

def simple_progress_bar(current, total, animal, cond, bar_length=20):
    '''simple progress bar for long running loops'''
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current >= .99*total else '\r'
    print(f'{animal} {cond} Progress: [{arrow}{padding}] {int(fraction*100)}%  ', end=ending)