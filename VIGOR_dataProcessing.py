# util functions for the project
import os
import re
import pandas as pd
import numpy as np
import copy
from itertools import groupby, chain
from scipy import stats
from scipy.signal import find_peaks
import pickle
import datetime
import fnmatch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as smooth

from VIGOR_plotting import *
import sessionlists

plt.style.use('./Figures/test.mplstyle')
PALETTE = {'RatF00': (0.55, 0.0, 0.0), 'RatF01': (0.8, 0.36, 0.36), 'RatF02': (1.0, 0.27, 0.0), 'RatF03': (.5, .5, .5),
           'RatM00': (0.0, 0.39, 0.0), 'RatM01': (0.13, 0.55, 0.13), 'RatM02': (0.2, 0.8, 0.2), 'RatM03': (.5, .5, .5)}


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


# replace first 0s in animal position (animal not found when cam init)
# if animal not found == camera edit, so replace with the first ok position
def fix_start_session(pos, edit):
    fixed = np.array(copy.deepcopy(pos))
    _edit = np.array(copy.deepcopy(edit))
    first_zero = next((i for i, x in enumerate(_edit) if not x), None)
    fixed[:first_zero] = pos[first_zero]
    _edit[:first_zero] = 0
    return fixed.flatten(), _edit.flatten()


# function to split lists --> used to split the raw X position array into
# smaller arrays (runs and stays). Later in the code we modify the
# array and change some values to 0, which will be used as cutting points.
def split_a_list_at_zeros(List):
    return [list(g) for k, g in groupby(List, key=lambda x:x != 0) if k]


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


# Old function to compute start of run and end of run boundaries
def extract_boundaries(data, dist, height=None):
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


# convert scale, convert i = 0 to 120 --> 60 to-60 which correspnds to the speed to the right (0 to 60) and to the left (0 to -60)
def convert_scale(number):
    old_min = 0
    old_max = 120
    new_max = -60
    new_min = 60
    return int(((number - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)


# compute mask to separate runs and stays based on speed
def filterspeed(dataPos, dataSpeed, threshold, dist):
    # dissociate runs from non runs, we want a cut off based on animal, speed. How to define this speed? If we plot the speed of the animal in function of the X position in the apparatus, so we can see that there is some blobs of speeds close to 0 and near the extremities of the treadmill, these are the ones that we want to define as non running speeds. With this function we want to compute the area of these points of data (higher density, this technique might not work when animals are not properly trained) in order to differentiate them.
    middle = dist/2
    xmin, xmax = 0, 120  # specify the x and y range of the window that we want to analyse
    ymin, ymax = -60, 60
    position = np.array(dataPos, dtype=float)  # data needs to be transformed to float perform the KDE
    speed = np.array(dataSpeed, dtype=float)
    X, Y = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]  # create 2D grid to compute KDE
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([position, speed])
    kernel = stats.gaussian_kde(values)  # compute KDE, this gives us an estimation of the point density. SLOW
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
def removeSplits_Mask(inputMask, inputPos, dist):
    correctedMask = [list(val) for key, val in groupby(inputMask, lambda x: x == True)]
    splitPos = []
    middle = (dist)/2
    count = [0, 0, 0, 0, 0, 0]
    start, end = 0, 0
    for elem in correctedMask:
        start = end
        end = start + len(elem)
        splitPos.append(inputPos[start:end])
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


# cuts session in bins
def bin_session(data_to_cut, data_template, bins):
    output = {}
    bincount = 0
    for timebin in bins:
        if timebin[0] == 0:
            start_of_bin = 0
        else:
            start_of_bin = int(np.where(data_template == timebin[0])[0])+1
        end_of_bin = int(np.where(data_template == timebin[1])[0])+1
        output[bincount] = data_to_cut[start_of_bin:end_of_bin]
        bincount += 1
    return output


# due to the way blocks are computed, some runs may have started in block[n] and ended in block [n+1], this function appends the end of the run to the previous block. See reCutBins.
def fixSplittedRunsMask(input_Binmask, blocks):
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
        dataSession[i] = list(chain.from_iterable(list(data.values())))
    return dataSession


# separate runs/stays * left/right + other variables into dicts
def extract_runSpeedBin(dataPos, dataSpeed, dataTime, dataLickR, dataLickL, openR, openL, mask, blocks, boundary, treadmillspeed, rewardProbaBlock):
    runs = {}
    stays = {}
    runs = {}
    stays = {}
    position, speed, time, running_Xs, idle_Xs, running_speedX, waiting_speedX, running_Time, waiting_Time = ({bin: [] for bin in range(0, len(blocks))} for _ in range(9))
    speedRunToRight, speedRunToLeft, XtrackRunToRight, XtrackRunToLeft, timeRunToRight, timeRunToLeft, timeStayInRight, timeStayInLeft, XtrackStayInRight, XtrackStayInLeft, TtrackStayInRight, TtrackStayInLeft, instantSpeedRight, instantSpeedLeft, maxSpeedRight, maxSpeedLeft, whenmaxSpeedRight, whenmaxSpeedLeft, wheremaxSpeedRight, wheremaxSpeedLeft, lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft = ({bin: [] for bin in range(0, len(blocks))} for _ in range(26))
    rewardedLeft, rewardedRight = ({bin: [] for bin in range(0, len(blocks))} for _ in range(2))

    for i in range(0, len(blocks)):
        position[i] = np.array(dataPos[i], dtype=float)
        speed[i] = np.array(dataSpeed[i], dtype=float)
        time[i] = np.array(dataTime[i], dtype=float)

        running_Xs[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(position[i], mask[i])]]
        idle_Xs[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(position[i], mask[i])]]
        running_speedX[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(speed[i], mask[i])]]
        waiting_speedX[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(speed[i], mask[i])]]
        running_Time[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(time[i], mask[i])]]
        waiting_Time[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(time[i], mask[i])]]

        stays[i] = [[e[0], e[1], e[2], e[3], e[4]] if [e[0], e[1], e[2]] != [None, None, None] else 0 for e in [[i, j, k, l, m] for i, j, k, l, m in zip(running_Xs[i], running_speedX[i], running_Time[i], dataLickR[i], dataLickL[i])]]
        runs[i] = [[e[0], e[1], e[2], e[3], e[4]] if [e[0], e[1], e[2]] != [None, None, None] else 0 for e in [[i, j, k, l, m] for i, j, k, l, m in zip(idle_Xs[i], waiting_speedX[i], waiting_Time[i], openL[i], openR[i])]]

        for run in split_a_list_at_zeros(runs[i]):
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

        for stay in split_a_list_at_zeros(stays[i]):
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
    return speedRunToRight, speedRunToLeft, XtrackRunToRight, XtrackRunToLeft, timeRunToRight, \
        timeRunToLeft, timeStayInRight, timeStayInLeft, XtrackStayInRight, XtrackStayInLeft, \
        TtrackStayInRight, TtrackStayInLeft, instantSpeedRight, instantSpeedLeft, \
        maxSpeedRight, maxSpeedLeft, whenmaxSpeedRight, whenmaxSpeedLeft, \
        wheremaxSpeedRight, wheremaxSpeedLeft, lick_arrivalRight, lick_drinkingRight, \
        lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft, rewardedRight, rewardedLeft


# in action sequence, cut full action sequence into corresponding blocks
def recut(data_to_cut, data_template):
    output = []
    start_of_bin = 0
    for i, _ in enumerate(data_template):
        end_of_bin = start_of_bin + len(data_template[i])
        output.append(data_to_cut[start_of_bin: end_of_bin])
        start_of_bin = end_of_bin
    return output


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


# much faster, mask is ok but then bug
def filterspeed2(dataPos, dataSpeed, threshold, dist):
    middle = dist/2
    xmin, xmax = 0, 120  # specify the x and y range of the window that we want to analyse
    ymin, ymax = -60, 60
    position = np.array(dataPos, dtype=float)  # data needs to be transformed to float perform the KDE
    speed = np.array(dataSpeed, dtype=float)

    xbins = np.linspace(xmin, xmax, xmax+1)
    ybins = np.linspace(ymin, ymax, ymax+1)

    heatmap, _, __ = np.histogram2d(dataPos, dataSpeed, bins=(xbins, ybins))

    hm = heatmap.T
    hm[hm < threshold] = False
    hm[hm >= threshold] = True
    plt.imshow(hm, aspect="auto")

    mask = np.zeros_like(position, dtype=bool)
    for line in range(ymin, ymax):
        pos = np.where(hm[line] == True)[0]
        if pos[pos < middle].size:
            low = pos[pos < middle][0]
            high = pos[pos < middle][-1]
            a = np.ma.masked_less(position, high)
            b = np.ma.masked_greater(position, low)
            c = np.ma.masked_less(speed, line + 0.5)
            d = np.ma.masked_greater(speed, line - 0.5)
            mask = np.logical_and(a.mask, b.mask)
            mask2 = np.logical_and(c.mask, d.mask)
            combiLeft = np.logical_and(mask, mask2)
            if not mask.size:
                mask = combiLeft
            else:
                mask = np.logical_xor(combiLeft, mask)
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
            if not mask.size:
                mask = combiRight
            else:
                mask = np.logical_xor(combiRight, mask)
    return ~mask


class ProcessData():
    def __init__(self, root, animal, session, buggedSessions, redoMask=False, redoFig=False):
        self.root = root
        self.animal = animal
        self.session = session
        self.lickBug, self.notfixed, self.F00lostTRACKlick, self.buggedRatSessions, self.boundariesBug, self.runstaysepbug = buggedSessions
        self.redoMask = redoMask
        self.redoFig = redoFig
        self.run()

    def run(self):
        self.figPath = f'{self.root}{os.sep}{self.animal}{os.sep}Experiments{os.sep}{self.session}{os.sep}Figures{os.sep}recapFIG{self.session}.png'
        self.maskpicklePath = f'{self.root}{os.sep}{self.animal}{os.sep}Experiments{os.sep}{self.session}{os.sep}Analysis{os.sep}mask.p'

        self.get_rat_colors()
        self.params = self.get_session_parameters()
        self.read_data_from_file()
        self.create_mask()
        self.extract_all_variables()
        if self.redoFig:
            self.plot_recap_figure()
        self.save_and_delete_variables()

    def get_rat_colors(self):
        '''
        This function defines the color and the marker of the rat
        for plots
        '''
        if fnmatch.fnmatch(self.animal, 'RatF*'):
            self.rat_markers = [PALETTE[self.animal], "$\u2640$"]
        elif fnmatch.fnmatch(self.animal, 'RatM*'):
            self.rat_markers = [PALETTE[self.animal], "$\u2642$"]
        elif fnmatch.fnmatch(self.animal, 'Rat00*'):
            self.rat_markers = [PALETTE[self.animal], "$\u2426$"]
        else:
            print("error, this is not a rat you got here")

    def get_session_parameters(self):
        '''
        This function reads the parameters of the session and returns a dictionary with the parameters
        Some parameters are computed from the raw data
        '''
        # compute the number of days since the last adlib
        lastDayadlib = str(datetime.datetime.strptime(str(read_params(self.root, self.animal, self.session, "lastDayadlib")), "%Y%m%d").date())
        stringmatch = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', self.session)
        experimentDay = str(datetime.datetime.strptime(stringmatch.group(), '%Y_%m_%d_%H_%M_%S'))
        _deD, _meD, _seD = int(experimentDay[0:4]), int(experimentDay[5:7]), int(experimentDay[8:10])
        _daL, _maL, _saL = int(lastDayadlib[0:4]), int(lastDayadlib[5:7]), int(lastDayadlib[8:10])
        daysSinceadlib = datetime.date(_deD, _meD, _seD) - datetime.date(_daL, _maL, _saL)

        # compute the real session duration
        realEnd = read_params(self.root, self.animal, self.session, "ClockStop")
        if str(realEnd) != 'None':
            startExpe = datetime.time(int(experimentDay[11:13]), int(experimentDay[14:16]), int(experimentDay[17:19]))
            endExpe = datetime.time(hour=int(str(realEnd)[0:2]), minute=int(str(realEnd)[2:4]), second=int(str(realEnd)[4:6]))
            realSessionDuration = datetime.datetime.combine(datetime.date(1, 1, 1), endExpe) - datetime.datetime.combine(datetime.date(1, 1, 1), startExpe)
        else:
            realSessionDuration = None

        # get block parameter from file
        blocklist = []
        for blockN in range(1, 13):  # Max 12 blocks, coded in LabView...
            # add block if  block >0 seconds then get data from file.
            # Data from behav_params as follows:
            # Block N°: // ON block Duration // OFF block duration // Repeat block // % reward ON // % reward OFF // Treadmill speed.
            if read_params(self.root, self.animal, self.session, "Block " + str(blockN), dataindex=-6, valueType=str) != 0:
                blocklist.append([read_params(self.root, self.animal, self.session, "Block " + str(blockN), dataindex=-6, valueType=str),
                                  read_params(self.root, self.animal, self.session, "Block " + str(blockN), dataindex=-5, valueType=str),
                                  read_params(self.root, self.animal, self.session, "Block " + str(blockN), dataindex=-4, valueType=str),
                                  read_params(self.root, self.animal, self.session, "Block " + str(blockN), dataindex=-3, valueType=str),
                                  read_params(self.root, self.animal, self.session, "Block " + str(blockN), dataindex=-2, valueType=str),
                                  read_params(self.root, self.animal, self.session, "Block " + str(blockN), dataindex=-1, valueType=str),
                                  blockN])
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

        # get size of water drop and weight of cup
        cupWeight = read_params(self.root, self.animal, self.session, "cupWeight", valueType=float)
        waterLeft = read_params(self.root, self.animal, self.session, "waterLeft", valueType=float)
        waterRight = read_params(self.root, self.animal, self.session, "waterRight", valueType=float)

        # build the session parameters dictionary
        # comment
        # lickthreshold: Labview 2021/07/06. Now uses the custom lickthreshold for each side. Useful when lickdata baseline
        # drifts and value is directly changed in LV. Only one session might be bugged, so this parameter is session specific.
        # Before, the default value (300) was used and modified manually during the analysis.
        session_params = {"sessionDuration": read_params(self.root, self.animal, self.session, "sessionDuration"),
                          "acqPer": read_params(self.root, self.animal, self.session, "acqPer"),
                          "waterLeft": round((waterLeft - cupWeight)/10*1000, 2),
                          "waterRight": round((waterRight - cupWeight)/10*1000, 2),
                          "treadmillDist": read_params(self.root, self.animal, self.session, "treadmillSize"),
                          "weight": read_params(self.root, self.animal, self.session, "ratWeight"),
                          "lastWeightadlib": read_params(self.root, self.animal, self.session, "ratWeightadlib"),
                          "lastDayadlib": read_params(self.root, self.animal, self.session, "lastDayadlib"),
                          "lickthresholdLeft": read_params(self.root, self.animal, self.session, "lickthresholdLeft"),
                          "lickthresholdRight": read_params(self.root, self.animal, self.session, "lickthresholdRight"),
                          "brainstatus": read_params(self.root, self.animal, self.session, "brainstatus", valueType="other"),
                          "boundaries": [],
                          "daysSinceadLib": daysSinceadlib.days,
                          "realSessionDuration": realSessionDuration,
                          "blocks": blocks,
                          "rewardP_ON": rewardP_ON,
                          "rewardP_OFF": rewardP_OFF,
                          "treadmillSpeed": treadmillSpeed,
                          "rewardProbaBlock": rewardProbaBlock,
                          }
        return session_params

    def read_data_from_file(self):
        '''
        Read data from the .position file and do some preprosessing
        '''
        # Read data from files
        file = self.root + os.sep + self.animal + os.sep + "Experiments" + os.sep + self.session + os.sep + self.session + ".position"
        extractTime = read_csv_pandas(file, Col=[3])
        extractPositionX = read_csv_pandas(file, Col=[4])
        extractPositionY = read_csv_pandas(file, Col=[5])
        extractLickLeft = read_csv_pandas(file, Col=[6])
        extractLickRight = read_csv_pandas(file, Col=[7])
        solenoid_ON_Left = read_csv_pandas(file, Col=[8])
        solenoid_ON_Right = read_csv_pandas(file, Col=[9])
        framebuffer = read_csv_pandas(file, Col=[10])
        cameraEdit = read_csv_pandas(file, Col=[11])
        self.framebuffer = framebuffer

        # Cut leftover data at the end of the session (e.g. session is 1800s long, data goes up
        # to 1820s because session has not been stopped properly/stopped manually, so we remove the extra 20s)
        rawTime = extractTime[extractTime <= self.params["sessionDuration"]]
        rawPositionX = extractPositionX[extractTime <= self.params["sessionDuration"]]
        rawPositionY = extractPositionY[extractTime <= self.params["sessionDuration"]]
        rawLickLeftX = extractLickLeft[extractTime <= self.params["sessionDuration"]]
        rawLickLeftY = extractLickLeft[extractTime <= self.params["sessionDuration"]]  # not needed, check
        rawLickRightX = extractLickRight[extractTime <= self.params["sessionDuration"]]
        rawLickRightY = extractLickRight[extractTime <= self.params["sessionDuration"]]  # not needed, check
        solenoid_ON_Left = solenoid_ON_Left[extractTime <= self.params["sessionDuration"]]
        solenoid_ON_Right = solenoid_ON_Right[extractTime <= self.params["sessionDuration"]]  # not needed, check
        cameraEdit = cameraEdit[extractTime <= self.params["sessionDuration"]]

        # convert data from px to cm
        rawPositionX, rawPositionY = datapx2cm(rawPositionX), datapx2cm(rawPositionY)
        rawSpeed = compute_speed(rawPositionX, rawTime)

        # usually rat is not found in the first few frames, so we replace Xposition by the first nonzero value
        # this is detected as a camera edit, so we fix that as well
        rawPositionX, cameraEdit = fix_start_session(rawPositionX, cameraEdit)
        rawPositionX = fixcamglitch(rawTime, rawPositionX, cameraEdit)

        # smoothing
        smoothPos, smoothSpeed = True, True
        sigmaPos, sigmaSpeed = 2, 2
        # seems to work, less: not smoothed enough,
        # more: too smoothed, not sure how to objectively compute an optimal value.
        if smoothPos is True:
            if smoothSpeed is True:
                rawPositionX = smooth(rawPositionX, sigmaPos)
                rawSpeed = smooth(compute_speed(rawPositionX, rawTime), sigmaSpeed)
            else:
                rawPositionX = smooth(rawPositionX, sigmaPos)

        ######################################################################################
        # LICKS AND WATER
        # Load lick data -- Licks == measure of conductance at the reward port.
        # Conductance is ____ and when lick, increase of conductance so ___|_|___, we define it as a lick
        # if it is above a threshold. But baseline value can randomly increase like this ___----,
        # so baseline can be above threshold, so false detections. -> compute moving median to get
        # the moving baseline (median, this way we eliminate the peaks in the calculation of the baseline)
        # and then compare with threshold. __|_|__---|---|----
        window = 200
        if self.params["lickthresholdLeft"] is None:
            self.params["lickthresholdLeft"] = 300
        if self.params["lickthresholdRight"] is None:
            self.params["lickthresholdRight"] = 300
        rawLickLeftX = [k if i-j >= self.params["lickthresholdLeft"] else 0 for i, j, k in
                        zip(rawLickLeftX, movinmedian(rawLickLeftX, window), rawPositionX)]
        rawLickRightX = [k if i-j >= self.params["lickthresholdRight"] else 0 for i, j, k in
                         zip(rawLickRightX, movinmedian(rawLickRightX, window), rawPositionX)]

        # Specify if a session has lick data problems, so we don't discard the whole session
        # (keep the run behavior, remove lick data)
        if all(v == 0 for v in rawLickLeftX):
            self.params["hasLick"] = False
        elif all(v == 0 for v in rawLickRightX):
            self.params["hasLick"] = False
        elif self.animal + " " + self.session in lickBug:
            self.params["hasLick"] = False
        else:
            self.params["hasLick"] = True

        # Water data. Drop size and volume rewarded. Compute drop size for each reward port. Determine
        # if drops are equal, or which one is bigger. Assign properties (e.g. line width for plots) accordingly.
        limitWater_diff = 5
        watL = round(self.params["waterLeft"], 1)  # print(round(self.params["waterLeft"], 1), "µL/drop")
        watR = round(self.params["waterRight"], 1)  # print(round(self.params["waterRight"], 1), "µL/drop")
        if watL-(watL*limitWater_diff/100) <= watR <= watL+(watL*limitWater_diff/100):
            water = ["Same Reward Size", "Same Reward Size", 2, 2]
        elif watL < watR:
            water = ["Small Reward", "Big Reward", 1, 5]
        elif watL > watR:
            water = ["Big Reward", "Small Reward", 5, 1]
        else:
            water = ["r", "r", 1, 1]

        # get a idea of where the rat is running/not running
        border = 5  # define arbitrary border
        self.leftBoundaryPeak, self.rightBoundaryPeak, self.kde = extract_boundaries(rawPositionX, self.params['treadmillDist'], height=0.001)

        # add to params and self
        self.params["boundaries"] = [self.rightBoundaryPeak - border, self.leftBoundaryPeak + border]
        self.rawPositionX, self.rawPositionY, self.rawSpeed, self.rawTime = rawPositionX, rawPositionY, rawSpeed, rawTime
        self.rawLickLeftX, self.rawLickRightX = rawLickLeftX, rawLickRightX
        self.solenoid_ON_Left, self.solenoid_ON_Right = solenoid_ON_Left, solenoid_ON_Right

    def create_mask(self):
        '''
        create run/not run mask based on animal position and speed
        animal is resting in the sides of the apparatus, and running in the middle
        animal is not running if it is in the sides and moving slowly
        1st method uses KDE to crearte the mask, slow (30s)
        2nd method uses a 2d histo to create the mask, fast (.3s)
        2nd method currently not used (mask ~ok, bugs downstream)
        '''
        if os.path.exists(self.maskpicklePath) and (not self.redoMask):
            self.binMask = get_from_pickle(self.root, self.animal, self.session, name="mask.p")
        else:
            if self.animal + " " + self.session in runstaysepbug:
                septhreshold = 0.0004
                # threshold 0.0004 seems to work ok for all TM distances. lower the thresh the bigger
                # the wait blob zone taken, which caused problems in 60cm configuration.
            else:
                septhreshold = 0.0002
            rawMask = filterspeed(self.rawPositionX, self.rawSpeed, septhreshold, self.params["treadmillDist"])
            # bound = np.min([self.kde(self.leftBoundaryPeak), self.kde(self.rightBoundaryPeak)])*len(self.rawPositionX)/4
            # rawMask2 = filterspeed2(self.rawPositionX, self.rawSpeed, bound, self.params["treadmillDist"])
            self.smoothMask = removeSplits_Mask(rawMask, self.rawPositionX, self.params["treadmillDist"])
            self.binMask = fixSplittedRunsMask(bin_session(self.smoothMask, self.rawTime, self.params["blocks"]), self.params["blocks"])

        self.smoothMask = stitch([self.binMask])[0]
        self.running_Xs = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(self.rawPositionX, self.smoothMask)]]
        self.idle_Xs = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(self.rawPositionX, self.smoothMask)]]
        self.speed_while_running = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(self.rawSpeed, self.smoothMask)]]
        self.speed_while_waiting = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(self.rawSpeed, self.smoothMask)]]
        self.binSpeed = reCutBins(self.rawSpeed, self.binMask)
        self.binTime = reCutBins(self.rawTime, self.binMask)
        self.binPositionX = reCutBins(self.rawPositionX, self.binMask)
        self.binPositionY = reCutBins(self.rawPositionY, self.binMask)
        self.binLickLeftX = reCutBins(self.rawLickLeftX, self.binMask)
        self.binLickRightX = reCutBins(self.rawLickRightX, self.binMask)
        self.binSolenoid_ON_Left = reCutBins(self.solenoid_ON_Left, self.binMask)
        self.binSolenoid_ON_Right = reCutBins(self.solenoid_ON_Right, self.binMask)

    def extract_all_variables(self):
        '''
        get all variables for the session
        speed
        idle
        lick
        etc
        '''
        # Extract all variables.
        (speedRunToRightBin, speedRunToLeftBin, XtrackRunToRightBin, XtrackRunToLeftBin,
            timeRunToRightBin, timeRunToLeftBin, timeStayInRightBin, timeStayInLeftBin,
            XtrackStayInRightBin, XtrackStayInLeftBin, TtrackStayInRightBin, TtrackStayInLeftBin,
            instantSpeedRightBin, instantSpeedLeftBin, maxSpeedRightBin, maxSpeedLeftBin,
            whenmaxSpeedRightBin, whenmaxSpeedLeftBin, wheremaxSpeedRightBin, wheremaxSpeedLeftBin,
            lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin,
            lick_arrivalLeftBin, lick_drinkingLeftBin, lick_waitLeftBin,
            rewardedRightBin, rewardedLeftBin) = extract_runSpeedBin(self.binPositionX, self.binSpeed, self.binTime,
                                                                     self.binLickRightX, self.binLickLeftX, self.binSolenoid_ON_Right, self.binSolenoid_ON_Left,
                                                                     self.binMask, self.params['blocks'], self.params["boundaries"],
                                                                     self.params["treadmillSpeed"], self.params['rewardProbaBlock'])

        (speedRunToRight, speedRunToLeft, XtrackRunToRight, XtrackRunToLeft,
            timeRunToRight, timeRunToLeft, timeStayInRight, timeStayInLeft,
            XtrackStayInRight, XtrackStayInLeft, TtrackStayInRight, TtrackStayInLeft,
            instantSpeedRight, instantSpeedLeft, maxSpeedRight, maxSpeedLeft,
            whenmaxSpeedRight, whenmaxSpeedLeft, wheremaxSpeedRight, wheremaxSpeedLeft,
            lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft,
            lick_drinkingLeft, lick_waitLeft, rewardedRight, rewardedLeft) = stitch(
                [speedRunToRightBin, speedRunToLeftBin, XtrackRunToRightBin, XtrackRunToLeftBin,
                 timeRunToRightBin, timeRunToLeftBin, timeStayInRightBin, timeStayInLeftBin,
                 XtrackStayInRightBin, XtrackStayInLeftBin, TtrackStayInRightBin, TtrackStayInLeftBin,
                 instantSpeedRightBin, instantSpeedLeftBin, maxSpeedRightBin, maxSpeedLeftBin,
                 whenmaxSpeedRightBin, whenmaxSpeedLeftBin, wheremaxSpeedRightBin, wheremaxSpeedLeftBin,
                 lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin, lick_arrivalLeftBin,
                 lick_drinkingLeftBin, lick_waitLeftBin, rewardedRightBin, rewardedLeftBin])

        nb_runs_to_rightBin, nb_runs_to_leftBin, nb_runsBin, total_trials= {}, {}, {}, 0
        for i in range(0, len(self.params['blocks'])):
            nb_runs_to_rightBin[i] = len(speedRunToRightBin[i])
            nb_runs_to_leftBin[i] = len(speedRunToLeftBin[i])
            nb_runsBin[i] = len(speedRunToRightBin[i]) + len(speedRunToLeftBin[i])
            total_trials += nb_runsBin[i]

        nb_rewardBlockLeft, nb_rewardBlockRight, nbWaterLeft, nbWaterRight = {}, {}, 0, 0
        for i in range(0, len(self.params['blocks'])):
            # split a list because in data file we have %open written along valve opening time
            # duration (same value multiple time), so we only take the first one, verify >threshold, ...
            nb_rewardBlockLeft[i] = sum([1 if t[0] <= self.params['rewardProbaBlock'][i] else 0 for
                                         t in split_a_list_at_zeros(self.binSolenoid_ON_Left[i])])
            nb_rewardBlockRight[i] = sum([1 if t[0] <= self.params['rewardProbaBlock'][i] else 0 for
                                         t in split_a_list_at_zeros(self.binSolenoid_ON_Right[i])])
        nbWaterLeft = sum(nb_rewardBlockLeft.values())
        nbWaterRight = sum(nb_rewardBlockRight.values())
        totalWater = round((nbWaterLeft * self.params["waterLeft"] + nbWaterRight * self.params["waterRight"])/1000, 2), 'mL'

        # compute total X distance moved during the session for each rat. maybe compute XY.
        totalDistance = sum(abs(np.diff(self.rawPositionX)))/100

        # sequences
        changes = np.argwhere(np.diff(self.smoothMask)).squeeze()
        full = []
        full.append(self.smoothMask[:changes[0]+1])
        for i in range(0, len(changes)-1):
            full.append(self.smoothMask[changes[i]+1:changes[i+1]+1])
        full.append(self.smoothMask[changes[-1]+1:])
        fulltime = recut(self.rawTime, full)
        openings = recut(self.solenoid_ON_Left + self.solenoid_ON_Right, full)
        positions = recut(self.rawPositionX, full)
        sequence = {}
        for item, (j, t, o, p) in enumerate(zip(full, fulltime, openings, positions)):
            proba = split_a_list_at_zeros(o)[0][0] if np.any(split_a_list_at_zeros(o)) else 100
            start_time = t[0]
            action = "run" if j[0] == True else "stay"
            reward = 1 if proba < self.params['rewardProbaBlock'][get_block(t[0])] else 0
            action_duration = t[-1] - t[0]
            avg_speed = (p[-1] - p[0])/(t[-1] - t[0]) if j[0] == True else "wait"
            sequence[item] = start_time, action, reward, action_duration, avg_speed

        self.XtrackRunToRight, self.XtrackRunToLeft = XtrackRunToRight, XtrackRunToLeft
        self.timeRunToRight, self.timeRunToLeft = timeRunToRight, timeRunToLeft
        self.speedRunToRight, self.speedRunToLeft = speedRunToRight, speedRunToLeft
        self.maxSpeedRight, self.maxSpeedLeft = maxSpeedRight, maxSpeedLeft
        self.instantSpeedRight, self.instantSpeedLeft = instantSpeedRight, instantSpeedLeft
        self.wheremaxSpeedRight, self.wheremaxSpeedLeft = wheremaxSpeedRight, wheremaxSpeedLeft
        self.whenmaxSpeedRight, self.whenmaxSpeedLeft = whenmaxSpeedRight, whenmaxSpeedLeft
        self.XtrackStayInRight, self.XtrackStayInLeft = XtrackStayInRight, XtrackStayInLeft
        self.TtrackStayInRight, self.TtrackStayInLeft = TtrackStayInRight, TtrackStayInLeft
        self.timeStayInRight, self.timeStayInLeft = timeStayInRight, timeStayInLeft
        self.lick_arrivalRight, self.lick_arrivalLeft = lick_arrivalRight, lick_arrivalLeft
        self.lick_drinkingRight, self.lick_drinkingLeft = lick_drinkingRight, lick_drinkingLeft
        self.lick_waitRight, self.lick_waitLeft = lick_waitRight, lick_waitLeft
        self.nb_runsBin = nb_runsBin
        self.speedRunToLeftBin, self.speedRunToRightBin = speedRunToLeftBin, speedRunToRightBin
        self.maxSpeedRightBin, self.maxSpeedLeftBin = maxSpeedRightBin, maxSpeedLeftBin
        self.timeStayInLeftBin, self.timeStayInRightBin = timeStayInLeftBin, timeStayInRightBin
        self.totalDistance, self.totalWater, self.total_trials = totalDistance, totalWater, total_trials
        self.timeRunToLeftBin, self.timeRunToRightBin = timeRunToLeftBin, timeRunToRightBin
        self.XtrackRunToLeftBin, self.XtrackRunToRightBin = XtrackRunToLeftBin, XtrackRunToRightBin
        self.instantSpeedLeftBin, self.instantSpeedRightBin = instantSpeedLeftBin, instantSpeedRightBin
        self.rewardedRightBin, self.rewardedLeftBin = rewardedRightBin, rewardedLeftBin
        self.sequence = sequence

    def save_and_delete_variables(self):
        '''
        pickle all variables and delete them
        '''
        save_as_pickle(self.root, self.params, self.animal, self.session, "params.p")
        save_as_pickle(self.root, self.binMask, self.animal, self.session, "mask.p")
        save_as_pickle(self.root, self.nb_runsBin, self.animal, self.session, "nbRuns.p")
        save_as_pickle(self.root, [self.totalDistance, self.totalWater, self.total_trials], self.animal, self.session, "misc.p")
        save_as_pickle(self.root, [self.speedRunToLeftBin, self.speedRunToRightBin], self.animal, self.session, "avgSpeed.p")
        save_as_pickle(self.root, [[[np.sum(np.diff(j)) for j in self.timeRunToLeftBin[i]]for i in range(0, len(self.params['blocks']))],
                                   [[np.sum(np.diff(j)) for j in self.timeRunToRightBin[i]]for i in range(0, len(self.params['blocks']))]], self.animal, self.session, "timeRun.p")
        save_as_pickle(self.root, [self.maxSpeedLeftBin, self.maxSpeedRightBin], self.animal, self.session, "maxSpeed.p")
        save_as_pickle(self.root, [self.timeStayInLeftBin, self.timeStayInRightBin], self.animal, self.session, "timeinZone.p")
        save_as_pickle(self.root, [self.XtrackRunToLeftBin, self.XtrackRunToRightBin], self.animal, self.session, "trackPos.p")
        save_as_pickle(self.root, [self.instantSpeedLeftBin, self.instantSpeedRightBin], self.animal, self.session, "trackSpeed.p")
        save_as_pickle(self.root, [self.timeRunToLeftBin, self.timeRunToRightBin], self.animal, self.session, "trackTime.p")
        save_as_pickle(self.root, [self.binLickLeftX, self.binLickRightX, self.binSolenoid_ON_Left, self.binSolenoid_ON_Right], self.animal, self.session, "lick_valves.p")
        save_as_pickle(self.root, [self.rewardedRightBin, self.rewardedLeftBin], self.animal, self.session, "rewarded.p")
        save_as_pickle(self.root, [self.TtrackStayInLeft, self.TtrackStayInRight], self.animal, self.session, "trackTimeinZone.p")
        save_as_pickle(self.root, self.sequence, self.animal, self.session, "sequence.p")

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # FLUSH
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Delete all data for this session
        flush = False
        if flush:
            del self.params
            del self.rawPositionX, self.rawPositionY, self.rawSpeed, self.rawTime
            del self.rawLickLeftX, self.rawLickRightX
            del self.solenoid_ON_Left, self.solenoid_ON_Right
            del self.leftBoundaryPeak, self.rightBoundaryPeak, self.kde

            del self.smoothMask, self.binMask, self.smoothMask
            del self.running_Xs, self.idle_Xs, self.speed_while_running, self.speed_while_waiting
            del self.binSpeed, self.binTime
            del self.binPositionX, self.binPositionY, self.binLickLeftX, self.binLickRightX
            del self.binSolenoid_ON_Left, self.binSolenoid_ON_Right

            del self.params, self.animal, self.session, self.root, self.binMask
            del self.XtrackRunToRight, self.XtrackRunToLeft, self.timeRunToRight, self.timeRunToLeft
            del self.speedRunToRight, self.speedRunToLeft, self.maxSpeedRight, self.maxSpeedLeft
            del self.instantSpeedRight, self.instantSpeedLeft, self.wheremaxSpeedRight, self.wheremaxSpeedLeft, self.whenmaxSpeedRight, self.whenmaxSpeedLeft
            del self.XtrackStayInRight, self.XtrackStayInLeft, self.TtrackStayInRight, self.TtrackStayInLeft
            del self.timeStayInRight, self.timeStayInLeft
            del self.lick_arrivalRight, self.lick_arrivalLeft, self.lick_drinkingRight, self.lick_drinkingLeft, self.lick_waitRight, self.lick_waitLeft
            del self.sequence
            del self.speedRunToLeftBin, self.speedRunToRightBin, self.maxSpeedRightBin, self.maxSpeedLeftBin
            del self.timeStayInLeftBin, self.timeStayInRightBin
            del self.totalDistance, self.totalWater, self.total_trials, self.nb_runsBin
            del self.timeRunToLeftBin, self.timeRunToRightBin, self.XtrackRunToLeftBin, self.XtrackRunToRightBin
            del self.instantSpeedLeftBin, self.instantSpeedRightBin, self.rewardedRightBin, self.rewardedLeftBin

    def plot_recap_figure(self):
        '''
        plot a recap figure with all the data
        '''
        fig = plt.figure(constrained_layout=False, figsize=(32, 42))
        fig.suptitle(self.session, y=0.9, fontsize=24)
        gs = fig.add_gridspec(75, 75)

        # position histogram
        ax00 = fig.add_subplot(gs[0:7, 0:4])
        plot_peak(ax00, self.rawPositionX, self.leftBoundaryPeak, self.rightBoundaryPeak, self.kde,
                  [0.05, 0], [0, 120], xyLabels=["Position (cm)", "%"])
        # position in session
        ax01 = fig.add_subplot(gs[0:7, 5:75])
        plot_BASEtrajectoryV2(ax01, self.rawTime, self.running_Xs, self.idle_Xs, self.rawLickLeftX, self.rawLickRightX,
                              self.params['rewardProbaBlock'], self.params['blocks'],
                              barplotaxes=[0, self.params['sessionDuration'], 50, 90, 0, 22, 10],
                              xyLabels=["Time (min)", " ", "Position (cm)", "", "", ""])
        ax01.plot([0, self.params['sessionDuration']], [self.params["boundaries"][0], self.params["boundaries"][0]], ":", color='k', alpha=0.5)
        ax01.plot([0, self.params['sessionDuration']], [self.params["boundaries"][1], self.params["boundaries"][1]], ":", color='k', alpha=0.5)

        # speed in session
        gs10 = gs[8:13, 0:75].subgridspec(2, 75)
        ax11 = fig.add_subplot(gs10[0, 5:75])
        ax12 = fig.add_subplot(gs10[1, 0:75])
        ax11.plot(self.rawTime, self.speed_while_running, color='dodgerblue')
        ax11.plot(self.rawTime, self.speed_while_waiting, color='orange')
        ax11.set_xlabel('time (s)')
        ax11.set_ylabel('speed (cm/s)')
        ax11.set_xlim(0, 3600)
        ax11.set_ylim(-200, 200)
        ax11.spines['top'].set_color("none")
        ax11.spines['right'].set_color("none")
        ax11.spines['left'].set_color("none")
        ax11.spines['bottom'].set_color("none")

        # speed per position
        ax12.scatter(self.rawPositionX, self.speed_while_running, color='dodgerblue', s=0.5)
        ax12.scatter(self.rawPositionX, self.speed_while_waiting, color='orange', s=0.5)
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

        # track of the rat
        ax20 = fig.add_subplot(gs[17:22, 0:10])
        plot_tracks(ax20, self.XtrackRunToRight, self.timeRunToRight, self.params["boundaries"],
                    xylim=[-0.1, 2, 0, 120], color=['paleturquoise', 'tomato'],
                    xyLabels=["Time (s)", "X Position (cm)"], title="Tracking runs to Right")
        ax21 = fig.add_subplot(gs[17:22, 15:25])
        plot_tracks(ax21, self.XtrackRunToLeft, self.timeRunToLeft, self.params["boundaries"],
                    xylim=[-0.1, 2, 0, 120], color=['darkcyan', 'darkred'],
                    xyLabels=["Time (s)", ""], title="Tracking runs to Left")

        # speed of the rat
        ax20 = fig.add_subplot(gs[17:22, 30:40])
        cumul_plot(ax20, self.speedRunToRight, self.speedRunToLeft, maxminstepbin=[0, 120, 1],
                   color=['paleturquoise', 'darkcyan', 'tomato', 'darkred'],
                   xyLabels=["Speed cm/s", "Cumulative Frequency Run Speed"],
                   title="Cumulative Plot Good Run Speed",
                   legend=["To Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "To Left:  " + str(self.params["waterLeft"]) + "µL/drop"])

        ax21 = fig.add_subplot(gs[17:22, 45:55])
        distribution_plot(ax21, self.speedRunToRight, self.speedRunToLeft, [0, 3, 120],
                          color=['paleturquoise', 'darkcyan', 'tomato', 'darkred'],
                          xyLabels=["Speed (cm/s)", "Direction of run", "To Right", "To Left"],
                          title="Distribution of All Run Speed",
                          legend=["To Right: Good Runs ", "To Left: Good Runs"])

        # time per frame
        gs23 = gs[15:22, 60:75].subgridspec(5, 2)
        ax231 = fig.add_subplot(gs23[0:2, 0:2])
        if len(self.framebuffer) != 0:
            ax231.set_title("NbBug/TotFrames: %s/%s = %.2f" % (sum(np.diff(self.framebuffer)-1),
                            len(self.framebuffer), sum(np.diff(self.framebuffer)-1)/len(self.framebuffer)))
        ax231.scatter(list(range(1, len(self.framebuffer))), [x-1 for x in np.diff(self.framebuffer)], s=5)
        ax231.set_xlabel("frame index")
        ax231.set_ylabel("dFrame -1 (0 is ok)")

        ax232 = fig.add_subplot(gs23[3:5, 0:2])
        ax232.set_title(self.params["realSessionDuration"])
        ax232.plot(np.diff(self.rawTime), label="data")
        ax232.plot(movinavg(np.diff(self.rawTime), 100), label="moving average")
        ax232.set_xlim(0, len(np.diff(self.rawTime)))
        ax232.set_ylim(0, 0.1)
        ax232.set_xlabel("frame index")
        ax232.set_ylabel("time per frame (s)")

        # max speed
        ax30 = fig.add_subplot(gs[25:30, 0:10])
        cumul_plot(ax30, self.maxSpeedRight, self.maxSpeedLeft, maxminstepbin=[0, 200, 1],
                   color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                   xyLabels=["Speed cm/s", "Cumulative Frequency MAX Run Speed"],
                   title="Cumulative Plot MAX Run Speed",
                   legend=["To Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "To Left:  " + str(self.params["waterLeft"]) + "µL/drop"])

        ax31 = fig.add_subplot(gs[25:30, 15:25])
        distribution_plot(ax31, self.maxSpeedRight, self.maxSpeedLeft, [0, 3, 200],
                          color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                          xyLabels=["Speed (cm/s)", "Direction of run", "To Right", "To Left"],
                          title="Distribution of MAX Run Speed",
                          legend=["To Right: Good Runs ", "To Left: Good Runs"])

        ax32 = fig.add_subplot(gs[25:30, 30:40])
        plot_speed(ax32, self.instantSpeedRight, self.timeRunToRight, [0, 0],
                   xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", "X Speed (cm/s)"],
                   title="To Right" + "\n" + str(self.params["waterRight"]) + "µL/drop")
        ax33 = fig.add_subplot(gs[25:30, 45:55])
        plot_speed(ax33, self.instantSpeedLeft, self.timeRunToLeft, [0, 0],
                   xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", ""],
                   title="To Left" + "\n" + str(self.params["waterLeft"]) + "µL/drop")
        ax34 = fig.add_subplot(gs[25:30, 60:70])
        plot_speed(ax34, self.instantSpeedRight + self.instantSpeedLeft,
                   self.timeRunToRight + self.timeRunToLeft, [0, 0],
                   xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", ""],
                   title=["Speed" + "\n" + " To left and to right"])

        # max speed howmuch/where/when
        ax40 = fig.add_subplot(gs[35:40, 0:8])
        cumul_plot(ax40, self.maxSpeedRight, self.maxSpeedLeft, maxminstepbin=[0, 200, 1],
                   color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                   xyLabels=["Speed cm/s", "Cumulative Frequency MAX Run Speed"],
                   title="Cumulative Plot MAX Run Speed",
                   legend=["To Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "To Left:  " + str(self.params["waterLeft"]) + "µL/drop"])

        ax41 = fig.add_subplot(gs[35:40, 12:23])
        distribution_plot(ax41, self.maxSpeedRight, self.maxSpeedLeft, [0, 3, 200],
                          color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                          xyLabels=["Speed (cm/s)", "Direction of run", "To Right", "To Left"],
                          title="Distribution of MAX Run Speed",
                          legend=["To Right", "To Left"])

        ax42 = fig.add_subplot(gs[35:40, 26:34])  # where maxspeed
        cumul_plot(ax42, self.wheremaxSpeedRight, self.wheremaxSpeedLeft, [0, 120, 1],
                   color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                   xyLabels=["Position maxSpeed reached (cm)", "Cumulative Frequency MAX runSpeed Position"],
                   title="CumulPlt MAXrunSpeed \nPosition from start of run",
                   legend=["To Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "To Left:  " + str(self.params["waterLeft"]) + "µL/drop"])

        ax43 = fig.add_subplot(gs[35:40, 38:49])
        distribution_plot(ax43, self.wheremaxSpeedRight, self.wheremaxSpeedLeft, [0, 2.5, 120],
                          color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                          xyLabels=["X Position (cm)", "Direction of run", "To Right", "To Left"],
                          title="Distr. MAXrunSpeed \nPosition from start of run",
                          legend=["To Right", "To Left"])

        ax44 = fig.add_subplot(gs[35:40, 52:60])  # where maxspeed
        cumul_plot(ax44, self.whenmaxSpeedRight, self.whenmaxSpeedLeft, [0, 2.5, 1],
                   color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                   xyLabels=["Time maxSpeed reached (cm)", "Cumulative Frequency MAX runSpeed Time"],
                   title="CumulPlt Time \nMAXrunSpeed from start of run",
                   legend=["To Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "To Left:  " + str(self.params["waterLeft"]) + "µL/drop"])

        ax45 = fig.add_subplot(gs[35:40, 64:75])
        distribution_plot(ax45, self.whenmaxSpeedRight, self.whenmaxSpeedLeft, [0, 2.5, 1],
                          color=['lightgreen', 'darkgreen', 'tomato', 'darkred'],
                          xyLabels=["Time MAX runSpeed reached (s)", "Direction of run", "To Right", "To Left"],
                          title="Distr. Time \nMAXrunSpeed from start of run",
                          legend=["To Right", "To Left"])

        # track in sides
        ax50 = fig.add_subplot(gs[45:50, 0:10])
        plot_tracks(ax50, self. XtrackStayInRight, self.TtrackStayInRight, self.params["boundaries"],
                    xylim=[-1, 10, self.params['treadmillDist']-40, self.params['treadmillDist']],
                    color=['moccasin', 'tomato'],
                    xyLabels=["Time (s)", "X Position (cm)"], title="Tracking in Right")

        ax51 = fig.add_subplot(gs[45:50, 15:25])
        plot_tracks(ax51, self.XtrackStayInLeft, self.TtrackStayInLeft, self.params["boundaries"],
                    xylim=[-1, 10, 0, 40], color=['darkorange', 'darkred'],
                    xyLabels=["Time (s)", ""], title="Tracking in Left")

        ax52 = fig.add_subplot(gs[45:50, 30:40])
        cumul_plot(ax52, self.timeStayInRight, self.timeStayInLeft, [0, 15, 1],
                   color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                   xyLabels=["Time in zone (s)", "Cumulative Frequency Time In Zone"],
                   title="Cumulative Plot Good Time In Zone",
                   legend=["In Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "In Left:  " + str(self.params["waterLeft"]) + "µL/drop"])

        ax53 = fig.add_subplot(gs[45:50, 45:60])
        distribution_plot(ax53, self.timeStayInRight, self.timeStayInLeft, [0, 3, 30],
                          color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                          xyLabels=["Time in zone (s)", "Zone", "In Right", "In Left"],
                          title="Distribution of All Time In Zone",
                          legend=["To Right", "To Left"])

        # lick data
        ax60 = fig.add_subplot(gs[55:60, 0:8])
        cumul_plot(ax60, self.lick_arrivalRight, self.lick_arrivalLeft, [0, 2, 1],
                   color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                   xyLabels=["Time (s)", "Cumulative Frequency"], title="Cumulative Plot preDrink Time",
                   legend=["In Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "In Left:  " + str(self.params["waterLeft"]) + "µL/drop"])

        ax61 = fig.add_subplot(gs[55:60, 12:23])
        distribution_plot(ax61, self.lick_arrivalRight, self.lick_arrivalLeft, [0, 3, 2],
                          color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                          xyLabels=["Time (s)", "Zone", "In Right", "In Left"],
                          title="Distribution preDrink Time",
                          legend=["In Right", "In Left"])

        ax62 = fig.add_subplot(gs[55:60, 26:34])
        cumul_plot(ax62, self.lick_drinkingRight, self.lick_drinkingLeft, [0, 4, 1],
                   color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                   xyLabels=["Time (s)", "Cumulative Frequency"], title="Cumulative Plot Drink Time",
                   legend=["In Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "In Left:  " + str(self.params["waterLeft"]) + "µL/drop"])
        ax63 = fig.add_subplot(gs[55:60, 38:49])
        distribution_plot(ax63, self.lick_drinkingRight, self.lick_drinkingLeft, [0, 3, 4],
                          color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                          xyLabels=["Time (s)", "Zone", "In Right", "In Left"],
                          title="Distribution of Drink Time",
                          legend=["In Right", "In Left"])

        ax64 = fig.add_subplot(gs[55:60, 52:60])
        cumul_plot(ax64, self.lick_waitRight, self.lick_waitLeft, [0, 10, 1],
                   color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                   xyLabels=["Time (s)", "Cumulative Frequency"], title="Cumulative Plot postDrink Time",
                   legend=["In Right: " + str(self.params["waterRight"]) + "µL/drop",
                           "In Left:  " + str(self.params["waterLeft"]) + "µL/drop"])
        ax65 = fig.add_subplot(gs[55:60, 64:75])
        distribution_plot(ax65, self.lick_waitRight, self.lick_waitLeft, [0, 3, 10],
                          color=['moccasin', 'darkorange', 'tomato', 'darkred'],
                          xyLabels=["Time (s)", "Zone", "In Right", "In Left"],
                          title="Distribution of postDrink Time",
                          legend=["In Right", "In Left"])

        if len(self.params['blocks']) > 1:
            stat = "Med. "
            blocks = self.params['blocks']
            ax70 = fig.add_subplot(gs[63:70, 0:9])
            plot_figBin(ax70, [self.nb_runsBin[i]/(int((blocks[i][1]-blocks[i][0])/60)) for i in range(0, len(blocks))],
                        self.params['rewardProbaBlock'], blocks,
                        barplotaxes=[0, self.params['sessionDuration']/60, 0, 25],
                        color='k', xyLabels=["Time (min)", "\u0023 runs / min"],
                        title="", stat=stat)

            ax72 = fig.add_subplot(gs[63:70, 20:29])
            plot_figBin(ax72, [self.speedRunToLeftBin[i] + self.speedRunToRightBin[i] for i in range(0, len(blocks))],
                        self.params['rewardProbaBlock'], blocks,
                        barplotaxes=[0, self.params['sessionDuration']/60, 0, 100],
                        color='dodgerblue', xyLabels=["Time (min)", "Avg. run speed (cm/s)"],
                        title="", scatter=True, stat=stat)

            ax74 = fig.add_subplot(gs[63:70, 40:49])
            plot_figBin(ax74, [self.maxSpeedRightBin[i] + self.maxSpeedLeftBin[i] for i in range(0, len(blocks))],
                        self.params['rewardProbaBlock'], blocks,
                        barplotaxes=[0, self.params['sessionDuration']/60, 0, 150],
                        color='red', xyLabels=["Time (min)", "Average max speed (cm/s)"],
                        title="", scatter=True, stat=stat)

            ax76 = fig.add_subplot(gs[63:70, 60:69])
            plot_figBin(ax76, [self.timeStayInLeftBin[i] + self.timeStayInRightBin[i] for i in range(0, len(blocks))],
                        self.params['rewardProbaBlock'], blocks,
                        barplotaxes=[0, self.params['sessionDuration']/60, 0, 25],
                        color='orange', xyLabels=["Time (min)", "Avg. time in sides (s)"],
                        title="", scatter=True, stat=stat)

            ax71 = fig.add_subplot(gs[63:70, 10:15])
            plot_figBinMean(ax71, [i/(int((self.params['blocks'][block][1]-self.params['blocks'][block][0])/60))
                                   for block, i in enumerate(poolByReward([self.nb_runsBin], self.params["rewardP_OFF"][0],
                                                                          self.params['blocks'], self.params['rewardProbaBlock']))],
                                  [i/(int((self.params['blocks'][block][1]-self.params['blocks'][block][0])/60))
                                   for block, i in enumerate(poolByReward([self.nb_runsBin], self.params["rewardP_ON"][0],
                                                                          self.params['blocks'], self.params['rewardProbaBlock']))],
                            color='k', ylim=(0, 25))

            ax73 = fig.add_subplot(gs[63:70, 30:35])
            plot_figBinMean(ax73, [np.mean(i) for i in poolByReward([self.speedRunToRightBin, self.speedRunToLeftBin], self.params["rewardP_OFF"][0],
                                                                    self.params['blocks'], self.params['rewardProbaBlock'])],
                                  [np.mean(i) for i in poolByReward([self.speedRunToRightBin, self.speedRunToLeftBin], self.params["rewardP_ON"][0],
                                                                    self.params['blocks'], self.params['rewardProbaBlock'])],
                            color='dodgerblue', ylim=(0, 100))

            ax75 = fig.add_subplot(gs[63:70, 50:55])
            plot_figBinMean(ax75, [np.mean(i) for i in poolByReward([self.maxSpeedRightBin, self.maxSpeedLeftBin], self.params["rewardP_OFF"][0],
                                                                    self.params['blocks'], self.params['rewardProbaBlock'])],
                                  [np.mean(i) for i in poolByReward([self.maxSpeedRightBin, self.maxSpeedLeftBin], self.params["rewardP_ON"][0],
                                                                    self.params['blocks'], self.params['rewardProbaBlock'])],
                            color='red', ylim=(0, 150))

            ax77 = fig.add_subplot(gs[63:70, 70:75])
            plot_figBinMean(ax77, [np.mean(i) for i in poolByReward([self.timeStayInRightBin, self.timeStayInLeftBin], self.params["rewardP_OFF"][0],
                                                                    self.params['blocks'], self.params['rewardProbaBlock'])],
                                  [np.mean(i) for i in poolByReward([self.timeStayInRightBin, self.timeStayInLeftBin], self.params["rewardP_ON"][0],
                                                                    self.params['blocks'], self.params['rewardProbaBlock'])],
                            color='orange', ylim=(0, 25))

        # %config InlineBackend.print_figure_kwargs = {'bbox_inches':None} #use % in notebook
        ax80 = fig.add_subplot(gs[73:74, 0:60])
        ax80.spines['top'].set_color("none")
        ax80.spines['right'].set_color("none")
        ax80.spines['left'].set_color("none")
        ax80.spines['bottom'].set_color("none")
        ax80.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax80.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        text = ''
        for k, v in self.params.items():
            text += f'{k}: {v} | '

        ax80.text(0, 0, str(text), wrap=True)

        save_sessionplot_as_png(self.root, self.animal, self.session,
                                f'recapFIG{self.session}.png', dpi='figure',
                                transparent=False, background='w')
        plt.close('all')

# if __name__ == "__main__":
#     # Define data path.
#     if platform.system()=='Linux':
#         root="/home/david/Desktop/DATA"
#         savePath="/home/david/Desktop/Save"
#     elif platform.system()=='Darwin':
#         root="/Users/tom/Desktop/DATA"
#         savePath="/Users/tom/Desktop/Save"
#     # if 'COLAB_GPU' in os.environ:
#     #     !gdown --id 1oxWJLF67TEifzQFgtUHIyhnEsS6AeQUW
#     #     !unzip -qq /content/code/datacopy.zip
#     #     root="/content/code/datacopy"
#     #     savePath="/content/Save"
#     #     print("I'm running on Colab")
#     print(f"Path to data is: {root}")

#     print(f"Current working directory: {os.getcwd()}")
#     print("Save Path: ", savePath)

#     sessionLists = pickle.load(open("picklejar/sessionLists.p", "rb"))
#     trainDist, dist60, dist60bis, dist90, dist90bis, dist120, dist120bis, TMtrain, TMrev20, TMrev10, TMrev2, TM2, TM10, TM20 = sessionLists

#     animalList = [os.path.basename(path) for path in sorted(glob.glob(root+"/Rat*"))]
#     animal = animalList[0]
#     sessionList = dist60[2:4]
#     sessionList = ['RatF00_2021_07_19_15_25_33']

#     p = ProcessData(root, animal, sessionList, buggedSessions, redoMask=False)
#     p.run()