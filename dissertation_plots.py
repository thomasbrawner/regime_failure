import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from sklearn.metrics import roc_curve, precision_recall_curve


def performance_plot(models, plotter=roc_curve, labels=None, fname=None):
    if labels is None: 
        labels = ['Model 1', 'Model 2']
    if plotter.func_name == 'roc_curve':
        xlabel, ylabel = 'False Positive Rate', 'True Positive Rate'
    elif plotter.func_name == 'precision_recall_curve':
        xlabel, ylabel = 'Recall', 'Precision'
    else:
        print('Warning: plot type not recognized, empty axis labels set')
        xlabel, ylabel = '', ''
    pdata = [] 
    for model in models:
        pdata.append(plotter(model.y, model.predictions))
    plt.plot(pdata[0][1], pdata[0][0], linestyle=':', label=labels[0])
    plt.plot(pdata[1][1], pdata[1][0], linestyle='--', label=labels[1])
    plt.xlabel(xlabel, labelpad=11)
    plt.ylabel(ylabel, labelpad=11)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if fname is not None: 
        plt.savefig(fname)
        plt.close() 
    else:
        plt.show() 


def separation_plot_data(y_true, y_prob): 
    pdata = pd.DataFrame([y_true, y_pred]).T
    pdata.columns = ['y', 'yhat']
    pdata = pdata.sort('yhat')
    pdata = pdata.reset_index(drop=True)
    events = pdata[pdata['y'] == 1]
    return pdata['yhat'].values, events.index.values


def separation_plot(models, alpha=0.8, labels=None, fname=None): 
    if labels is None:
        labels = ['Model 1', 'Model 2']
    pdata = []
    for model in models: 
        pdata.append(separation_plot_data(model.y, model.predictions))
    plt.figure(figsize=(12, (3 * len(models)))
    for row in xrange(1, len(models) + 1): 
        ax = plt.subplot(len(models), 1, row)
        ax.plot(pdata[0], '-')
        for event in pdata[1]:
            ax.axvline(x=event, linewidth=0.5, linestyle='-', color='r', alpha=alpha)
        ax.set_ylim([0, 1])
        ax.ylabel(label[row - 1])
    if fname is not None:
        plt.savefig(fname)
        plt.close() 
    else: 
        plt.show() 


def boxplot_estimates(ests, names, ignore=None, fname=None):
    if ignore is not None: 
        for factor in ignore:
            p = re.compile(factor)
            [names.remove(m) for m in filter(p.match, names)]
        ests = ests[:, :len(names)]
    data = pd.DataFrame(ests, columns=names)
    data = pd.melt(data)
    sns.boxplot(x='value', y='variable', data=data, color='0.50')
    plt.axvline(x=0, linestyle=':', c='k')
    plt.xlabel('Estimate'); plt.ylabel('')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
        plt.close() 
    else: 
        plt.show() 
