
# You'll be able to import this by using 

# from fariness_metrics import fair_metrics

# fair_metrics(y_actual, y_pred_prob, y_pred_binary, X_test, protected_group_name, adv_val, disadv_val)

# Pages 20 & 21

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

def fair_metrics(y_actual, y_pred_prob, y_pred_binary, X_test, protected_group_name, adv_val, disadv_val):
    """
    Fairness performance metrics for a model to compare advantageous and disadvantageous groups of a protected variable.
    
    Parameters
    ----------
    
    :param y_actual: Actual binary outcome
    :param y_pred_prob: Predicted probabilities
    :param y_pred_binary: Predicted binary outcome
    :param X_test: X_test data
    :param protected_group_name: Sensitive feature 
    :param adv_val: Privileged value of protected label
    :param disadv_val: Unprivileged value of protected label
    :return: roc, avg precision, Eq of Opportunity, Equalised Odds, Precision/Predictive Parity, 
             Demographic Parity, Avg Odds, Diff, Predictive Equality, Treatment Equality
    
    Examples
    --------
    
    fairness_metrics=[fair_metrics(y_test, y_pred_prob, y_pred, X_test, choice, adv_val, disadv_val)]
    
    """
    
    tn_adv, fp_adv, fn_adv, tp_adv = confusion_matrix(
        y_actual[X_test[protected_group_name] == adv_val],
        y_pred_binary[X_test[protected_group_name] == adv_val]
    ).ravel()
    
    tn_disadv, fp_disadv, fn_disadv, tp_disadv = confusion_matrix(
        y_actual[X_test[protected_group_name] == disadv_val],
        y_pred_binary[X_test[protected_group_name] == disadv_val]
    ).ravel()
    
    # Receiver operating characteristic
    roc_adv = roc_auc_score(
        y_actual[X_test[protected_group_name] == adv_val],
        y_pred_prob[X_test[protected_group_name] == adv_val]
    )
    
    roc_disadv = roc_auc_score(
        y_actual[X_test[protected_group_name] == disadv_val],
        y_pred_prob[X_test[protected_group_name] == disadv_val]
    )
    
    roc_diff = abs(roc_disadv - roc_adv)
    
    # Average precision score
    ps_adv = average_precision_score(
        y_actual[X_test[protected_group_name] == adv_val],
        y_pred_prob[X_test[protected_group_name] == adv_val]
    )
                                     
    ps_disadv = average_precision_score(
        y_actual[X_test[protected_group_name] == disadv_val],
        y_pred_prob[X_test[protected_group_name] == disadv_val]
    )

    ps_diff = abs(ps_disadv - ps_adv)
    
    # Equal Opportunity - advantageous and disadvantageous groups have equal FNR
    FNR_adv = fn_adv / (fn_adv + tp_adv)
    FNR_disadv = fn_disadv / (fn_disadv + tp_disadv)
    EOpp_diff = abs(FNR_disadv - FNR_adv)
    
    # Predictive equality - advantageous and disadvantageous groups have equal FPR
    FPR_adv = fp_adv / (fp_adv + tn_adv)
    FPR_disadv = fp_disadv / (fp_disadv + tn_disadv)
    pred_eq_diff = abs(FPR_disadv - FPR_adv)
    
    # Equalised Odds - advantageous and disadvantageous groups have equal TPR + FPR
    TPR_adv = tp_adv / (tp_adv + fn_adv)
    TPR_disadv = tp_disadv / (tp_disadv + fn_disadv)
    EOdds_diff = abs((TPR_disadv + FPR_disadv) - (TPR_adv + FPR_adv))
    
    # Predictive Parity - advantageous and disadvantageous groups have equal PPV/Precision (TP/TP+FP)
    prec_adv = tp_adv / (tp_adv + fp_adv)
    prec_disadv = tp_disadv / (tp_disadv + fp_disadv)
    prec_diff = abs(prec_disadv - prec_adv)
    
    # Demographic Parity - ratio of (instances with favorable prediction) / (total instances)
    demo_parity_adv = (tp_adv + fp_adv) / (tn_adv + fp_adv + fn_adv + tp_adv)
    demo_parity_disadv = (tp_disadv + fp_disadv) / (tn_disadv + fp_disadv + fn_disadv + tp_disadv)
    demo_parity_diff = abs(demo_parity_disadv - demo_parity_adv)
    
    # Average of Difference in FPR and TPR for advantageous and disadvantageous groups
    AOD = 0.5 * ((FPR_disadv - FPR_adv) + (TPR_disadv - TPR_adv))
    
    # Treatment Equality - advantageous and disadvantageous groups have equal ratio of FN/FP
    TE_adv = fn_adv / fp_adv
    TE_disadv = fn_disadv / fp_disadv
    TE_diff = abs(TE_disadv - TE_adv)
    
    return [
        ('AUC', roc_diff), ('Avg PrecScore', ps_diff), ('Equal Opps', EOpp_diff),
        ('PredEq', pred_eq_diff), ('Equal Odds', EOdds_diff), ('PredParity', prec_diff),
        ('DemoParity', demo_parity_diff), ('AOD', abs(AOD)), ('TEq', TE_diff)
    ]
