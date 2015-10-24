function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;

for epsilon = min(pval):stepsize:max(pval)
    tp = 0;
    fp = 0;
    fn = 0;
    for i = 1:size(yval,1)
        if (pval(i) < epsilon) && (yval(i) == 1)
            tp += 1;
        elseif (pval(i) < epsilon) && (yval(i) == 0)
            fp += 1;
        elseif (pval(i) >= epsilon) && (yval(i) == 1)
            fn += 1;
        end
    end

    if (tp + fp != 0) || (tp + fn != 0)
        prec = tp / (tp + fp);
        rec = tp / (tp + fn);

        F1 = 2 * prec * rec / (prec + rec);
    end
    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
