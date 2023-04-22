# util functions for the project
import os
import pandas as pd
import numpy as np
import copy
from itertools import groupby, chain
from scipy import stats
from scipy.signal import find_peaks
import pickle


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
                while output_Binmask[i][0] == True:
                    output_Binmask[i-1] = np.append(output_Binmask[i-1], output_Binmask[i][0])
                    output_Binmask[i] = np.delete(output_Binmask[i], 0)
            if output_Binmask[i-1][-1] == False and output_Binmask[i][0] == False:  # print(i, "case2")
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
