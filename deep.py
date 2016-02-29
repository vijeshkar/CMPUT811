#! /usr/bin/env python

import sys
import logging  as log

# Import required libraries. Fail if any is missing.
try:
    import numpy    as np
    import scipy    as sp
    import theanets as tn
    import theautil as tu
except ImportError as e:
    print "FATAL: Missing libraries."
    print e
    sys.exit(1)

# Try to get matplotlib. It is not strictly required.
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Specify the format of the data.
DTYPE = [("a", "<f4"), ("b", "<f4"), ("c", "|S6")]

# Max updates for training.
MAX_UPDATES = 1500

# A handy way to convert known classes to numbers and viceversa.
CLASSES = {
    'red': 0,
    'blue': 1,
    'yellow': 2
}

CLASS_NUMS = {
    0: 'red',
    1: 'blue',
    2: 'yellow'
}

CLASS2NUM = lambda c : CLASSES[c]
NUM2CLASS = lambda n : CLASS_NUMS[n]

# Debug variables.
PLOT = True
LOG_LEVEL = log.WARNING # It should run faster if it doesn't log everything.
XKCD = False # Have some fun with plots!
PLOT_TO_FILE = True

###
# Plot the input dataset.
###
def plot_dataset(data, labels):
    x = [data[i][0] for i in xrange(data.shape[0])]
    y = [data[i][1] for i in xrange(data.shape[0])]
    fig = plt.figure("Training data")
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, c = labels)
    ax1.set_aspect(1.0 / ax1.get_data_ratio())

    if not PLOT_TO_FILE:
        plt.show()
    else:
        fig.savefig("training_data_set.png")

###
# Plot test data with actual classes.
###
def plot_test_input(test, labels):
    x = [test[i][0] for i in xrange(test.shape[0])]
    y = [test[i][1] for i in xrange(test.shape[0])]

    fig = plt.figure("Test data")
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, c = labels)
    ax1.set_aspect(1.0 / ax1.get_data_ratio())

    if not PLOT_TO_FILE:
        plt.show()
    else:
        fig.savefig("test_data_set.png")

###
# Plot the results of a test.
###
def plot_test(test, labels, network):
    net_name = network.replace(',', '-').replace('[', '').replace(']', '')

    x = [test[i][0] for i in xrange(test.shape[0])]
    y = [test[i][1] for i in xrange(test.shape[0])]
    fig = plt.figure("Classified data" + network)
    ax1 = fig.add_subplot(111)
    train = ax1.scatter(x, y, c = labels)
    ax1.set_aspect(1.0 / ax1.get_data_ratio())

    if not PLOT_TO_FILE:
        plt.show()
    else:
        fig.savefig("test" + net_name + "_data_set.png")

###
# Save test data to file.
###
def save_test(test, labels, network):
    file_name = "test-" + network.replace(',', '-').replace('[', '').replace(']', '') + ".csv"

    x = [test[i][0] for i in xrange(test.shape[0])]
    y = [test[i][1] for i in xrange(test.shape[0])]
    out_test = np.transpose(np.array([x, y, labels]))

    np.savetxt(file_name, out_test, fmt= "%s,%s,%s,")
    print "Test saved to " + file_name

##
# Print confusion matrix and accuracy for a test.
##
def print_matrix(classified, real):
    matrix_1 = "|         | %7s | %7s | %7s |"
    matrix_2 = "| %7s | % 7d | % 7d | % 7d |"
    matrix_3 = "| %7s | % 7d | % 7d | % 7d |"
    matrix_4 = "| %7s | % 7d | % 7d | % 7d |"

    tp_red = sum([(1 if (classified[i] == real[i] and real[i] == 'red') else 0) for i in xrange(real.shape[0])])
    tp_blue = sum([(1 if (classified[i] == real[i] and real[i] == 'blue') else 0) for i in xrange(real.shape[0])])
    tp_yellow = sum([(1 if (classified[i] == real[i] and real[i] == 'yellow') else 0) for i in xrange(real.shape[0])])
    tp = float(tp_red + tp_blue + tp_yellow)

    fn_red_as_blue = sum([(1 if (real[i] == 'red' and classified[i] == 'blue') else 0) for i in xrange(real.shape[0])])
    fn_red_as_yellow = sum([(1 if (real[i] == 'red' and classified[i] == 'yellow') else 0) for i in xrange(real.shape[0])])
    fn_red = fn_red_as_blue + fn_red_as_yellow
    fn_blue_as_red = sum([(1 if (real[i] == 'blue' and classified[i] == 'red') else 0) for i in xrange(real.shape[0])])
    fn_blue_as_yellow = sum([(1 if (real[i] == 'blue' and classified[i] == 'yellow') else 0) for i in xrange(real.shape[0])])
    fn_blue = fn_blue_as_red + fn_blue_as_yellow
    fn_yellow_as_red = sum([(1 if (real[i] == 'yellow' and classified[i] == 'red') else 0) for i in xrange(real.shape[0])])
    fn_yellow_as_blue = sum([(1 if (real[i] == 'yellow' and classified[i] == 'blue') else 0) for i in xrange(real.shape[0])])
    fn_yellow = fn_yellow_as_red + fn_yellow_as_blue
    fn = float(fn_red + fn_blue + fn_yellow)

    tn_red = sum([(1 if (real[i] != 'red' and classified[i] != 'red') else 0) for i in xrange(real.shape[0])])
    tn_blue = sum([(1 if (real[i] != 'blue' and classified[i] != 'blue') else 0) for i in xrange(real.shape[0])])
    tn_yellow = sum([(1 if (real[i] != 'yellow' and classified[i] != 'yellow') else 0) for i in xrange(real.shape[0])])
    tn = float(tn_red + tn_blue + tn_yellow)

    fp_red = sum([(1 if (real[i] != 'red' and classified[i] == 'red') else 0) for i in xrange(real.shape[0])])
    fp_blue = sum([(1 if (real[i] != 'blue' and classified[i] == 'blue') else 0) for i in xrange(real.shape[0])])
    fp_yellow = sum([(1 if (real[i] != 'yellow' and classified[i] == 'yellow') else 0) for i in xrange(real.shape[0])])
    fp = float(fp_red + fp_blue + fp_yellow)

    print
    print "CONFUSION MATRIX: "
    print matrix_1 % ('red', 'blue', 'yellow')
    print matrix_2 % ('red', tp_red, fn_red_as_blue, fn_red_as_yellow)
    print matrix_3 % ('blue', fn_blue_as_red, tp_blue, fn_blue_as_yellow)
    print matrix_4 % ('yellow', fn_yellow_as_red, fn_yellow_as_blue, tp_yellow)
    print
    
    print "         | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) |"
    print " %7s | % 19d | % 19d | % 20d | % 20d |" % ('red', tp_red, tn_red, fp_red, fn_red)
    print " %7s | % 19d | % 19d | % 20d | % 20d |" % ('blue', tp_blue, tn_blue, fp_blue, fn_blue)
    print " %7s | % 19d | % 19d | % 20d | % 20d |" % ('yellow', tp_yellow, tn_yellow, fp_yellow, fn_yellow)

    tp_rate = tp / (tp + fn)
    fn_rate = fn / (fn + tp)
    fp_rate = fp / (fp + tn)
    tn_rate = tn / (tn + fp)

    print
    print "TP rate: " + str(tp_rate)
    print "TN rate: " + str(tn_rate)
    print "FN rate: " + str(fn_rate)
    print "FP rate: " + str(fp_rate)
    print
    print "ACC: " + str(tp / float(real.shape[0]))

###
# Run a test with a given deep network and plot the results if able.
###
def test_net(cnet, train, valid, test, test_real, network):
    cnet.train(train, valid, algo = 'layerwise', patience = 1, max_updates = MAX_UPDATES)
    cnet.train(train, valid, algo = 'rprop', patience = 1, max_updates = MAX_UPDATES)

    test_l = cnet.classify(test)
    test_t = np.array([NUM2CLASS(test_l[i]) for i in xrange(len(test_l))])
    print "%s / %s correctly classified instances" % (sum(test_t == test_real), test_t.shape[0])

    if PLOT and plt is not None:
        if XKCD:
            with plt.xkcd():
                plot_test(test, test_t, network)
        else:
            plot_test(test, test_t, network)

    save_test(test, test_t, network)

    print_matrix(test_t, test_real)

###
# Script main function
###
def main(data_file, test_file):
    try:
        # Load the data.
        _data = np.loadtxt(data_file, delimiter = ',', usecols = range(3), dtype = DTYPE)
        _test = np.loadtxt(test_file, delimiter = ',', usecols = range(3), dtype = DTYPE)
    except Exception:
        print "Could not load the data in " + data_file
        return 1

    # Separate the training dataset from it's labels.
    data = np.array([(_data[i][0], _data[i][1]) for i in xrange(_data.shape[0])])
    labels_t = np.array([_data[i][2] for i in xrange(_data.shape[0])])
    labels_n = np.array([CLASS2NUM(labels_t[i]) for i in xrange(labels_t.shape[0])]).astype(np.int32)

    # Separate the test dataset from it's labels. The labels are stored for checking later.
    test = np.array([(_test[i][0], _test[i][1]) for i in xrange(_test.shape[0])])
    test_real = np.array([_test[i][2] for i in xrange(_test.shape[0])])

    # Plot the data if needed and able to.
    if PLOT and plt is not None:
        if XKCD:
            with plt.xkcd():
                plot_dataset(data, labels_t)
                plot_test_input(test, test_real)
        else:
            plot_dataset(data, labels_t)
            plot_test_input(test, test_real)

    # Create the classifiers.
    cnet_1 = tn.Classifier([2, 4, 3])
    cnet_2 = tn.Classifier([2, 8, 4, 3])
    cnet_3 = tn.Classifier([2, 8, 6, 4, 3])

    # Shuffle and split the training data and labels.
    tu.joint_shuffle(data, labels_n)
    train, valid = tu.split_validation(90, data, labels_n)

    print "*******************************************************************"
    print "* TESTING CLASSIFIER: 2, 4, 3                                     *"
    print "*******************************************************************"

    test_net(cnet_1, train, valid, test, test_real, "[2,4,3]")

    print "*******************************************************************"
    print "* TESTING CLASSIFIER: 2, 8, 4, 3                                  *"
    print "*******************************************************************"

    test_net(cnet_2, train, valid, test, test_real, "[2,8,4,3]")

    print "*******************************************************************"
    print "* TESTING CLASSIFIER: 2, 8, 6, 4, 3                               *"
    print "*******************************************************************"

    test_net(cnet_3, train, valid, test, test_real, "[2,8,6,4,3]")

    return 0

###
# Script's entry point.
# Set up things and call main.
###
if __name__ == "__main__":
    log.basicConfig(stream = sys.stderr, level = LOG_LEVEL)

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        if len(sys.argv) > 2:
            test_file = sys.argv[2]
        else:
            test_file = "test_data.csv"            
    else:
        data_file = "data.csv"
        test_file = "test_data.csv"

    rc = main(data_file, test_file)
    sys.exit(rc)
