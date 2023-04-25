import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd


# conversion
def inch2cm(value): return value / 2.54
def cm2inch(value): return value * 2.54


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
    return (list)


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


# 2022_05_04 LV: added brain status of the animal (normal, lesion, cno, saline) in behav.params.
# This is a fix to consign it in antecedent sessions. I think it fixed them all.
def FIXwrite_params(root, animal, session):
    # animal = "RatF02"
    # for session in sorted(matchsession(animal, lesionrev20+lesionrev10+lesionrev2+lesion2+lesion10+lesion20)): #lesiontrain+lesion60+lesion90+lesion120
    #     FIXwrite_params(root, animal, session)
    behav = root + os.sep+animal + os.sep+"Experiments" + os.sep + session + os.sep + session + ".behav_param"
    if not os.path.exists(behav):
        print("No file %s" % behav)
    alreadywritten = False
    with open(behav, "r") as f:
        for line in f:
            if "brainstatus" in line:
                alreadywritten = True
    if not alreadywritten:
        with open(behav, "a") as f:
            f.write("\nbrainstatus normal")


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
        except Exception:
            print("error loading pickle")
            pass
    else:
        print("no pickle found")
        return None


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


def dirty_acceleration(array):
    return [np.diff(a)/0.04 for a in array]


# in action sequence, get block number based on beginning of action
def get_block(t_0):
    block = None
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
    return block


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

############################################
# copy only test.p files for all sessions
# compress and upload to google drive
# add animal, session = 'RatM01', 'RatM01_2021_07_22_17_14_48'  to the list
# import fnmatch
# from os.path import isdir, join
# from shutil import copytree, rmtree

# def include_patterns(*patterns):
#     def _ignore_patterns(path, all_names):
#         # Determine names which match one or more patterns (that shouldn't be
#         # ignored).
#         keep = (name for pattern in patterns for name in fnmatch.filter(all_names, pattern))
#         # Ignore file names which *didn't* match any of the patterns given that
#         # aren't directory names.
#         dir_names = (name for name in all_names if isdir(join(path, name)))
#         return set(all_names) - set(keep) - set(dir_names)
#     return _ignore_patterns


# src = "/home/david/Desktop/DATA"
# dst = "/home/david/Desktop/testcopy"
# copytree(src, dst, ignore=include_patterns('test.p'))
############################################