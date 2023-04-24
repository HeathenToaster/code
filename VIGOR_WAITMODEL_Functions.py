import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from scipy import stats
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from VIGOR_dataProcessing import *

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
            temp_data[time_bin][animal] = {k: [] for k in meankeys(targetlist)}
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
            temp_data[bin][animal] = {key: {k: [] for k in meankeys(targetlist)} for key in matchsession(animal, sessionList)}
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
    '''
    read position file and plot animal trajectory
    '''
    if ax is None:
        ax = plt.gca()
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
                        xyLabels=[" ", " ", " ", " "], title="", scatter=False, ax=None):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    if ax is None:
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


# cut the full sequence dict in blocks
def bin_seq(seq):
    prevblock = 0
    index = 0
    binseq = {k: {} for k in [_ for _ in range(0, 12)]}
    for i in range(0, len(seq)):
        if get_block(seq[i][0]) != prevblock:
            index = i  # if change block (next block) store action# to reset first action of next block to 0
        binseq[get_block(seq[i][0])][i-index] = seq[i]
        prevblock = get_block(seq[i][0])
    return binseq


# raster of (non)rewarded trials, reward average selection, and idle time distribution plots
def plot_rewards(data, avg, memsize=3, ax=None, filter=[0, 3600]):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    input = bin_seq(data)

    # get the number of runs per block to init vars
    c = np.zeros(12)
    for i in range(12):
        n = 0
        for a in range(len(input[i])):
            if input[i][a][1] == "run":
                n += 1
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

    cols = np.array(['r', 'w', 'g'])
    cmap = ListedColormap(colors=cols)
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

    # ax.scatter(Xlast, Ylast, s=200, marker='x', c=last_rewards_from_previous_block,
    # cmap=cmap, vmin=0, vmax=1, edgecolors=edges, linewidths=1, alpha=0.35)
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
    ax.set_title('Reward sequence in example session')

    def _get_waiting_times_idx(data, memsize=3):
        """get waiting times idx from data"""
        waiting_times = {k: [] for k in meankeys(generate_targetList(seq_len=memsize)[::-1])}
        idx = 0
        for i in range(len(data)):
            if data[i][1] == 'stay':
                try:
                    avg_rwd = round(np.mean([data[i-n][2] for n in range(1, (memsize*2)+1, 2)]), 2)
                    waiting_times[avg_rwd].append(idx)
                except:  # put the first n waits in rwd=1 (because we don't have the previous n runs to compute the average reward)
                    waiting_times[1].append(idx)
                idx += 1
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
        if cc[0] <= res < cc[1]:
            idx, idy = 0, int(res-cc[0])
        if cc[1] <= res < cc[2]:
            idx, idy = 1, int(res-cc[1])
        if cc[2] <= res < cc[3]:
            idx, idy = 2, int(res-cc[2])
        if cc[3] <= res < cc[4]:
            idx, idy = 3, int(res-cc[3])
        if cc[4] <= res < cc[5]:
            idx, idy = 4, int(res-cc[4])
        if cc[5] <= res < cc[6]:
            idx, idy = 5, int(res-cc[5])
        if cc[6] <= res < cc[7]:
            idx, idy = 6, int(res-cc[6])
        if cc[7] <= res < cc[8]:
            idx, idy = 7, int(res-cc[7])
        if cc[8] <= res < cc[9]:
            idx, idy = 8, int(res-cc[8])
        if cc[9] <= res < cc[10]:
            idx, idy = 9, int(res-cc[9])
        if cc[10] <= res < cc[11]:
            idx, idy = 10, int(res-cc[10])
        if cc[11] <= res < cc[12]:
            idx, idy = 11, int(res-cc[11])
        return idx, idy

    for r in res:
        idx, idy = _convert_res(r)
        didx, didy = _convert_res(r-memsize+1)

        if filter[0] <= times[idx, idy] <= filter[1]:
            timeres.append(times[idx, idy])  # 2D array index for time of the end of the sequence in the data
            dtimeres.append(times[didx, didy])  # 2D array index for time of the start of the sequence in the data
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
                   weights=np.ones(len(nextwait)) / len(nextwait) * 100,
                   label=label)
        ax[0].set_title(f"Idle time distribution after {avg}\nrewards obtained in 0-10 min")
        ax[0].set_xlabel("Idle time (s)")
        ax[0].set_ylabel("PDF")
        ax[0].set_xlim(0, 25)
        ax[0].set_ylim(0, 1.1)

        ax[1].hist(sorted(nextwait)[::-1], bins=bins, histtype='step', color=color, lw=2,
                   density=True,
                   weights=np.ones(len(nextwait)) / len(nextwait) * 100,
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
    waiting_times = {k: [] for k in meankeys(generate_targetList(seq_len=memsize)[::-1])}
    for i in range(len(data)):
        if data[i][1] == 'stay':
            if filter[0] <= data[i][0] <= filter[1] and data[i][3] != 0:
                if data[i][3] < toolong:  # filter out
                    try:
                        avg_rwd = round(np.mean([data[i-n][2] for n in range(1, (memsize*2)+1, 2)]), 2)
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
        if ax is None:
            ax = plt.gca()
        waits = np.asarray(waits)

        bins = np.linspace(0, waits.max(), int(max(waits)))
        ydata, xdata, _ = ax.hist(waits, bins=bins,
                                  color=color, alpha=1, zorder=1,
                                  density=True,  # weights=np.ones_like(waits) / len(waits),
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


def plot_DDMexample(mean, std, A, t0, N=100, title=''):
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
    N = 250
    t0 = 2
    std = 1
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
    x[x < 0] = 1e-10
    arg = 2 * np.pi * x ** 3
    res = alpha / np.sqrt(arg) * np.exp(-((alpha-gamma * x) ** 2) / (2 * x))
    return np.array(res, dtype=np.float64)


def Wald_cdf(x, alpha, theta, gamma):
    """Wald cdf"""
    # from https://github.com/mark-hurlstone/RT-Distrib-Fit
    x = x - theta
    x[x < 0] = 1e-10
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
    res = minimize(crit, params_init, args=x, bounds=((0, None), (0, 1e-8), (0, None)))
    return res.x, res.fun


def genWaldSamples(N, alpha, gamma, maximum=500):
    """generate Wald samples"""
    # 230x faster than drawfromDDM (pyDDM)
    # based on https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    x = np.linspace(1e-8, maximum, maximum*100)

    def p(x, alpha, gamma):
        return alpha / np.sqrt(2 * np.pi * x ** 3) * np.exp(-((alpha-gamma * x) ** 2) / (2 * x))

    def normalization(x, alpha, gamma):
        return simps(p(x, alpha, gamma), x)

    pdf = p(x, alpha, gamma)/normalization(x, alpha, gamma)
    cdf = np.cumsum(pdf)
    cdf /= max(cdf)

    u = np.random.uniform(0, 1, int(N))
    interp_function = interp1d(cdf, x)
    samples = interp_function(u)
    return samples


def example_wald_fit(mean, std, A, t0, N=100, ax=None, color='k'):
    """example of fitting Wald distribution"""
    if ax is None:
        ax = plt.gca()
    waits = genWaldSamples(N, A, mean)
    bins = np.linspace(0, waits.max(), int(max(waits)))
    ydata, xdata, _ = ax.hist(waits, bins=bins,
                              color=color, alpha=.5, zorder=1, 
                              density=True, # weights=np.ones_like(waits) / len(waits),
                              histtype="step", lw=2, cumulative=-1, label=f'N={N} simulated samples')

    x = np.linspace(0.01, 500, 10000)
    xdata = xdata[:-1]

    # fittime = time.time()
    (alpha, theta, gamma), lossWald = wald_fit(waits)
    ax.plot(x, 1-Wald_cdf(x, alpha, theta, gamma), color=color, lw=2, zorder=4, label=f'best fit')
    ydatapdf, xdatapdf, _ = ax.hist(waits, bins=bins, alpha=.0, zorder=1, density=True, histtype="step",)

    ax.set_xlim(1, 500)
    ax.set_ylim(.001, 1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('log Wait time')
    ax.set_ylabel('log 1-CDF')
    if mean == 0.1:
        ax.legend()

    return alpha, theta, gamma, lossWald


def plot_color_line(ax, x, y, z, cmap='viridis', vmin=None, vmax=None, alpha=1, linewidth=1, linestyle='-', zorder=1):
    """plot line with color based on z values"""
    from matplotlib.collections import LineCollection
    color = np.abs(np.array(z, dtype=np.float64))
    if vmin is None:
        vmin = np.nanmin(color)
    if vmax is None:
        vmax = np.nanmax(color)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
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
            _gamma = GAMMA[bin, avg]  # if GAMMA[bin, avg] > 0 else 1e-8
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


# params = a, t, g, a', t', g', a'', t'', g''
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

    res = minimize(f, params_init, args=(data, [N_bins, N_avg], N_params),
                   bounds=((0, None), (0, 1e-8), (0, None),
                   alpha_t_bounds, (0, 1e-8), gamma_t_bounds,
                   alpha_R_bounds, (0, 1e-8), gamma_R_bounds))
    return res.x, res.fun


################################################


def plot_parameter_evolutionIdleTime(p, axs=None, N_bins=6, N_avg=4):

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
    if ax is None:
        ax = plt.gca()
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
                    y = np.max([np.mean(data1) + 2*np.std(data1), np.mean(data2) + 2*np.std(data2)]) + c
                    ax[idx].plot((i, j), (y, y), color='k')
                    ax[idx].scatter((i+j)/2, y+.1, color='k', marker=r'$\ast$')
                    c += 0.1


def dict_to_xticklabels(d):
    """convert dict keys to xticklabels for ablation plots"""
    allkeys = list(d.keys())
    conv = lambda x: "-" if x else "+"
    result = [r"$\alpha'$"+"\n"+r"$\gamma'$"+"\n"+r"$\alpha''$"+"\n"+r"$\gamma''$"]
    for i in allkeys:
        result.append(f'{chr(10).join([conv(j) for j in i])}')
    return result


def test_all_keys_between_themselves(losses, keys, ax=None):
    """dirty stats to test all conditions against each other, but with keys"""
    if ax is None:
        ax = plt.gca()
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
