import numpy as np
from .utils.metrics import metricor
from .analysis.robustness_eval import generate_curve
        
def get_metrics(score, labels, metric='all', version='opt', slidingWindow=None, thre=250):
    metrics = {}
    if metric == 'vus':
        grader = metricor()
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)

        metrics['VUS_ROC'] = VUS_ROC
        metrics['VUS_PR'] = VUS_PR

        return metrics

    elif metric == 'range_auc':
        grader = metricor()
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        
        metrics['R_AUC_ROC'] = R_AUC_ROC
        metrics['R_AUC_PR'] = R_AUC_PR

        return metrics

    elif metric == 'auc':
        
        grader = metricor()
        AUC_ROC = grader.metric_new_auc(labels, score, plot_ROC=False)
        _, _, AUC_PR = grader.metric_PR(labels, score)

        metrics['AUC_ROC'] = AUC_ROC
        metrics['AUC_PR'] = AUC_PR

        return metrics
    
    else:
        from .basic_metrics import basic_metricor

        grader = metricor()
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)
        grader = basic_metricor()
        AUC_ROC, Precision, Recall, F, _, _, _, _, _, Precision_at_k = grader.metric_new(labels, score, plot_ROC=False)

        # Selectively include only desired metrics
        metrics['Precision'] = Precision
        metrics['Recall'] = Recall
        metrics['F'] = F
        metrics['VUS_ROC'] = VUS_ROC
        metrics['VUS_PR'] = VUS_PR

        if metric == 'all':
            return metrics
        else:
            return metrics[metric]
