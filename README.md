<h2 align="center"> Parasitic Capacitance and Risistance Predictor Based on GNN </h2>
** This is GPU version with mini-batch training of GNN4RC **

# We change the training strategy to unifiy training

# We merge 3 datasets into a dataset list

# TODO:

    ## original -- cuda1 pid=8182 9-14 21:28
    ## settings of regressors
    ## layer number of GNN
    ## remove the feature normalization
    ## remove weight values in loss_r -- nouse
    ## train and test a single dataset -- good with classification bad with regression
    ## classification only -- cuda2 9-14 12:18 pid:28721
                              cuda1 9-15 14:02 ultra8T classification pid:23964
    ## loss backward seperatively

    ## undersampling in training set -- pid:12846
    ## remove classification stage -- cuda1 9-20 14:04 pid:19394
    ## 