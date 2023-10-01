import numpy as np
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import matplotlib.pyplot as plt
import fnmatch


from VIGOR_utils import *
from VIGOR_plotting import *
from VIGOR_MODELS_Functions import *

animalList = ['RatF00', 'RatF01', 'RatF02', 'RatM00', 'RatM01', 'RatM02', 
              'RatF30', 'RatF31', 'RatF32', 'RatF33', 'RatM30', 'RatM31', 'RatM32', 
              'RatF40', 'RatF41', 'RatF42', 'RatM40', 'RatM41', 'RatM42', 'RatM43', 
                'RatF50', 'RatF51', 'RatF52', 'RatM50', 'RatM51', 'RatM52', 'RatM53', 'RatM54'
                ]
conds = ["60", "90", "120", "20", "10", "2", "rev10", "rev20"]



rat_markers = {}
brainstatus = {'RatF00': 'normal', 'RatF01': 'normal', 'RatF02': 'normal',
               'RatM00': 'normal', 'RatM01': 'normal', 'RatM02': 'normal',
            #    'RatF20': 'thcre', 'RatF21': 'thcre', 'RatF22': 'thcre',
            #    'RatM20': 'thcre', 'RatM21': 'thcre', 'RatM22': 'thcre',
               'RatF30': 'DLS', 'RatF31': 'DLS', 'RatF32': 'normal', 'RatF33': 'normal',
               'RatM30': 'DLS', 'RatM31': 'normal', 'RatM32': 'normal',
               'RatF40': 'DLS', 'RatF41': 'DMS', 'RatF42': 'normal',
               'RatM40': 'normal', 'RatM41': 'DLS', 'RatM42': 'DMS', 'RatM43': 'normal', 

                'RatF50': 'DMS', 'RatF51': 'DLS', 'RatF52': 'DLS',
               'RatM50': 'DMS', 'RatM51': 'DMS', 'RatM52': 'DLS', 'RatM53': 'normal', 'RatM54': 'normal'
               }

intact_map = plt.cm.get_cmap('winter')
lesion_map = plt.cm.get_cmap('autumn')

# markers = {'normal': 'o', 'thcre': 'd', 'lesion': 'x', 'biglesion': 'X'}
# lines = {'normal': '-', 'thcre': '--', 'lesion': ':', 'biglesion': '-.'}

markers = {'normal': 'o', 'DLS': 'X', 'DMS': 'x'}
lines = {'normal': '-', 'DLS': '-', 'DMS': '-'}
colormaps = {'normal': intact_map, 'DLS': lesion_map, 'DMS': lesion_map}

# RATS
animalList = ['RatF00', 'RatF01', 'RatF02', 'RatM00', 'RatM01', 'RatM02', 
              'RatF30', 'RatF31', 'RatF32', 'RatF33', 'RatM30', 'RatM31', 'RatM32', 
              'RatF40', 'RatF41', 'RatF42', 'RatM40', 'RatM41', 'RatM42', 'RatM43', 
                'RatF50', 'RatF51', 'RatF52', 'RatM50', 'RatM51', 'RatM52', 'RatM53', 'RatM54'
                ]
intact = ['RatF00', 'RatF01', 'RatF02', 'RatM00', 'RatM01', 'RatM02', 
            'RatF32', 'RatF33', 'RatM31', 'RatM32', 'RatF42', 'RatM40', 'RatM43', 'RatM53', 'RatM54']
    

for index, animal in enumerate(animalList):
    if fnmatch.fnmatch(animal, 'RatF*'):
        rat_markers[animal]=[colormaps[brainstatus[animal]](index/len(animalList)), 'd', lines[brainstatus[animal]]]
    elif fnmatch.fnmatch(animal, 'RatM*'):
        rat_markers[animal]=[colormaps[brainstatus[animal]](index/len(animalList)), 's', lines[brainstatus[animal]]]
    elif fnmatch.fnmatch(animal, 'Rat00*'):
        rat_markers[animal]=[(0.0, 0.0, 0.0), "$\u2426$",]
    else:
        print("error, this is not a rat you got here")




def compute_ICC(var, animalList=animalList, bootstrap=False, n_samples=10000):
    expected_values = {cond: np.mean([var[animal][cond] for animal in animalList]) for cond in conds}
    individual_intercepts = {}
    remaining_residuals = {}
    x = np.arange(len(conds))

    for animal in animalList:
        # compute expected value for each condition
        y = np.array([var[animal][cond] for cond in conds])
        y_expected = y - np.array([expected_values[cond] for cond in conds])

        # compute intercept for each animal
        X = np.ones((len(x), 1))
        coefficients = np.linalg.lstsq(X, y_expected.reshape(-1, 1), rcond=None)[0]
        intercept = coefficients[0][0]
        individual_intercepts[animal] = intercept

        # compute residuals for each animal
        y_corrected = y_expected - individual_intercepts[animal]
        remaining_residuals[animal] = np.var(y_corrected)

    # population ICC = variance of intercepts / (variance of intercepts + mean of residuals)
    alp = np.var(list(individual_intercepts.values()))
    eps = np.mean(list(remaining_residuals.values()))
    ICC_pop = alp / (alp + eps)

    # individual ICC = variance of intercepts / (variance of intercepts + residuals)
    ICC_indiv = {animal: alp / (alp + remaining_residuals[animal]) for animal in animalList}

    # confidence interval 
    lower_bound = None
    upper_bound = None
    ICC_bootstrap = None

    if bootstrap:
        # create bootstrap samples from estimated variance and residuals
        # compute ICC for each sample, compute confidence interval
        ICC_bootstrap = np.zeros(n_samples)
        samples = generate_ICC_bootstrap_samples(var, alp, eps, n=n_samples, animalList=animalList)
        for i in range(n_samples):
            ICC_bootstrap[i] = compute_ICC(samples[i], animalList=animalList)[0]
        
        lower_bound = np.percentile(ICC_bootstrap, 2.5)
        upper_bound = np.percentile(ICC_bootstrap, 97.5)

    return ICC_pop, ICC_indiv, [lower_bound, upper_bound, ICC_bootstrap]
        
def generate_ICC_bootstrap_samples(var, intercept_variance, residual_variance, n=1000, animalList=animalList):
    samples = []
    for i in range(n):
        sample = {}
        for animal in animalList:
            rand1 = np.random.normal(0, np.sqrt(intercept_variance))
            sample[animal] = {}
            for cond in conds:
                rand2 = np.random.normal(0, np.sqrt(residual_variance))
                sample[animal][cond] = var[animal][cond] + rand1 + rand2
        samples.append(sample)
    return samples


def mock_dataset(n_subjects=animalList, n_measurements=["60", "90", "120", "rev20", "rev10", "2", "10", "20"], noise=0):
    np.random.seed(2007)
    data = {}
    expected_value = np.arange(len(n_measurements))/10+ np.random.rand(len(n_measurements))/5
    expected_value *= 0
    individual_intercept = np.random.rand(len(n_subjects))

    for i, animal in enumerate(n_subjects):
        data[animal] = {}
        for j, cond in enumerate(n_measurements):
            data[animal][cond] = expected_value[j] + individual_intercept[i] + np.random.rand() * noise
    return data

def explain_ICC(noise=0, ax=None):
    if ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(3, 2), gridspec_kw={'width_ratios': [2, 1]})

    explanation_animals = ['RatF00', 'RatF01', 'RatF02', 'RatM00', 'RatM01', 'RatM02']

    data = mock_dataset(noise=noise, n_subjects=explanation_animals)
    pop_ICC, indiv_ICC, (conf_bottom, conf_top, ICC_bootstrap) = compute_ICC(data, animalList=explanation_animals, bootstrap=True)

    ymax = 0
    for animal in explanation_animals:

        axs[0].plot(np.arange(len(conds)), [data[animal][cond] for cond in conds],)
        temp_ymax = np.max([data[animal][cond] for cond in conds])
        if temp_ymax > ymax:
            ymax = temp_ymax
        
        axs[1].scatter(0, indiv_ICC[animal], s=5)

        axs[1].plot([.075, .125], [pop_ICC, pop_ICC], color='gray', lw=1, zorder=1)
        axs[1].scatter(0.1, pop_ICC, color='gray', s=2.5, zorder=1)

        violin_parts = axs[1].violinplot(positions=[.1], 
                        dataset=[ICC_bootstrap],
                        widths=.05, showextrema=False, 
                        quantiles=[0.025, 0.975])
        
        for vp in violin_parts['bodies']:
            vp.set_facecolor('lightgray')
            vp.set_edgecolor('lightgray')
            vp.set_linewidth(0)
            vp.set_alpha(1)
            vp.set_zorder(0)
        # for vp in violin_parts['cquantiles']:
        violin_parts['cquantiles'].set_facecolor('gray')
        violin_parts['cquantiles'].set_edgecolor('gray')
        violin_parts['cquantiles'].set_linewidth(.5)
        violin_parts['cquantiles'].set_alpha(.25)
        violin_parts['cquantiles'].set_zorder(1)
        
        for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
            try:
                vp = violin_parts[partname]
                vp.set_edgecolor('gray')
                vp.set_linewidth(0)
                vp.set_alpha(0)
            except:
                pass

    axs[0].set_ylabel("Value (a.u.)")
    axs[0].set_xticks(np.arange(len(conds)))
    axs[0].set_xlabel("Measurement #")
    axs[0].set_xlim(0, 7)
    axs[0].set_ylim(0, 1.1*ymax)
    space_axes(axs[0], x_ratio_left=.1, x_ratio_right=.1)
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])


    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, 0.1)
    axs[1].set_xticks([0, 0.1])
    axs[1].set_xticklabels([r"$\mathrm{ICC_{ind}}$", r"$\mathrm{ICC_{pop}}$"])
    axs[1].set_ylabel("Repeatability")
    space_axes(axs[1], x_ratio_left=.5, x_ratio_right=.5)

    axs[1].set_yticks([0, 0.5, 0.75, .9, 1])
    # axs[1].set_yticklabels('')
    axs[1].set_yticks([0.25, 0.625, 0.825, .95], minor=True)
    axs[1].set_yticklabels(["Poor", "Moderate", "Good", "Excellent"], minor=True)
    axs[1].tick_params(which='minor', length=0)


def compute_intercept(var, animalList=animalList, conds=conds):

    x = np.arange(len(conds))
    expected_values = {cond: np.mean([var[animal][cond] for animal in animalList]) for cond in conds}
    individual_intercepts = {}

    for animal in animalList:
        # compute expected value for each condition
        y = np.array([var[animal][cond] for cond in conds])
        y_expected = y - np.array([expected_values[cond] for cond in conds])

        # compute intercept for each animal
        X = np.ones((len(x), 1))
        coefficients = np.linalg.lstsq(X, y_expected.reshape(-1, 1), rcond=None)[0]
        intercept = coefficients[0][0]
        individual_intercepts[animal] = intercept

    return individual_intercepts


def confidence_ellipse(x, y, ax=None, n_std=2.0, color='k'):
    '''This is from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # contour
    ellipse_contour = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        linewidth=1, color=color, fill=False, alpha=.8, zorder=1)
    # fill
    ellipse_fill = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        linewidth=1, color=color, fill=True, alpha=0.1, zorder=0)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
                        .rotate_deg(45) \
                        .scale(scale_x, scale_y) \
                        .translate(mean_x, mean_y)

    ellipse_contour.set_transform(transf + ax.transData)
    ellipse_fill.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse_contour), ax.add_patch(ellipse_fill)


def PCA_individuals_plot(score, labels, explained_variance_ratio=['', ''], ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    x = score[:,0]  # PC1
    y = score[:,1]  # PC2

    x_males, y_males = [], []
    x_females, y_females = [], []

    for animal in labels:
        ax.scatter(x[labels==animal], y[labels==animal], 
                    c=[rat_markers[animal][0] for _ in range(len(x[labels==animal]))], 
                    marker=rat_markers[animal][1], s=2, zorder=10)

        if 'M' in animal:
            x_males.append(x[labels==animal])
            y_males.append(y[labels==animal])
        else:
            x_females.append(x[labels==animal])
            y_females.append(y[labels==animal])


    ax.axhline(0, color='gray', linestyle='--', linewidth=.5, zorder=0)
    ax.axvline(0, color='gray', linestyle='--', linewidth=.5, zorder=0)

    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')


    confidence_ellipse(np.array(x_males).flatten(), np.array(y_males).flatten(), color='g', ax=ax)
    confidence_ellipse(np.array(x_females).flatten(), np.array(y_females).flatten(), color='r', ax=ax)    

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    space_axes(ax, x_ratio_right=0)


def PCA_variables_plot(coeff, names='', explained_variance_ratio=['', ''], ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    assert len(coeff) == len(names)

    for i, var in enumerate(names):
        ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], 
                color='k', linestyle='-', linewidth=.5, width=0.001, head_width=0.025, zorder=1)

        offset = 1.25
        x_offset, y_offset = 0, 0

        # # if two variables are close to each other, we don't want their names to overlap
        # for j in range(i+1, len(variables)):

        #         if np.abs(coeff[i, 0] - coeff[j, 0]) < 0.1 and np.abs(coeff[i, 1] - coeff[j, 1]) < 0.01:
        #             print(i, j , np.abs(coeff[i, 0] - coeff[j, 0]) , np.abs(coeff[i, 1] - coeff[j, 1]))
        #             if np.abs(coeff[i, 0] - coeff[j, 0]) < np.abs(coeff[i, 1] - coeff[j, 1]):
        #                 x_offset = .1
        #             else:
        #                 y_offset = .1

        # heck it, just do it manually
        # if i == 0:
        #     y_offset = .05
        # elif i == 3:
        #     y_offset = -.05
        # elif i == 6:
        #     x_offset = .025
        #     y_offset = .025
        # elif i == 8:
        #     x_offset = -.05
        # elif i == 11:
        #     x_offset = .05

        ax.text(coeff[i, 0]*offset+x_offset, coeff[i, 1]*offset+y_offset, var, 
                color='k', ha='center', va='center', fontsize=5)

    # plot circle
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 1
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, color='gray', linewidth=.5, zorder=0)

    ax.axhline(0, xmin=-1, xmax=1, color='gray', linestyle='--', linewidth=.5, zorder=0)
    ax.axvline(0, ymin=-1, ymax=1, color='gray', linestyle='--', linewidth=.5, zorder=0)

    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1-1/30, 1+1/30)
    space_axes(ax)
    ax.spines['left'].set_bounds(-1, 1)