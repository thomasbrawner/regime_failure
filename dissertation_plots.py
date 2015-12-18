import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 


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


def auc_pr_curve(y_true, y_pred): 
    # area under the precision-recall curve 
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision) 


