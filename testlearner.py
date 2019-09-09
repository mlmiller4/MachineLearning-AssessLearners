import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import math  		   	  			    		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import sys
import pandas as pd
import matplotlib.pyplot as plt
  		   	  			    		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			    		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		   	  			    		  		  		    	 		 		   		 		  
        print "Usage: python testlearner.py <filename>"  		   	  			    		  		  		    	 		 		   		 		  
        sys.exit(1)  		   	  			    		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])

    # Read CSV into a dataframe using pandas and convert into ndarray
    # This works for the Istanbul.csv file
    df = pd.read_csv(sys.argv[1], sep=',')
    df = df.drop(df.columns[[0]], axis=1)
    data = df.as_matrix()

    #print(data.shape)
    #data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    #print testX.shape
    #print testY.shape

    """ 
    Part 1: Test DTlearner with leaf sizes from 1 to 50 without shuffling
    -----------------------------------------------------------------"""
    insample_rmse = []
    outsample_rmse = []

    for i in range(1, 51):
        learner = dt.DTLearner(leaf_size=i, verbose=False)

        learner.addEvidence(trainX, trainY) # train it
        #print learner.author()

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        insample_rmse.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        outsample_rmse.append(rmse)


    x = np.arange(1,51)
    fig, ax = plt.subplots()

    ax.plot(x, insample_rmse, label="In sample RMSE")
    ax.plot(x, outsample_rmse, label="Out of Sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
           title="RMSE vs. Leaf Size for Decision Tree Learner \n with Unshuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels =  ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    #plt.show()
    fig.savefig('DT_unshuffled.png')
    plt.close()
    '''--------------------------------------------------------------------------------------------------'''

    '''
    Test DT Learner with leaf sizes from 1 to 50 with shuffling of the training and testing datasets for
    each round
    ----------------------------------------------------------------------------------------------------'''
    insample_shuffled_rmse = []
    outsample_shuffled_rmse = []

    for i in range(1, 51):

        # Shuffle data
        np.random.shuffle(trainX)
        np.random.shuffle(trainY)
        np.random.shuffle(testX)
        np.random.shuffle(testY)

        # create a learner and train it
        #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        #learner = rt.RTLearner(leaf_size=1, verbose=False)
        #learner = bl.BagLearner(learner = dt.DTLearner, kwargs={"leaf_size":1}, bags=20, boost=False, verbose=False)
        #learner = it.InsaneLearner(verbose=False)

        learner.addEvidence(trainX, trainY) # train it
        #print learner.author()

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        insample_shuffled_rmse.append(rmse)


        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        outsample_shuffled_rmse.append(rmse)

    x = np.arange(1,51)
    fig, ax = plt.subplots()

    ax.plot(x, insample_shuffled_rmse, label="In sample RMSE")
    ax.plot(x, outsample_shuffled_rmse, label="Out of Sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
           title="RMSE vs. Leaf Size for Decision Tree Learner \n with Shuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels =  ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)

    #plt.show()
    fig.savefig('DT_Shuffled.png')
    plt.close()
    '''--------------------------------------------------------------------------------'''

    '''
    Part 2: Test whether bagging can reduce or eliminate overfitting with respect to leaf size -
    Use DTlearner with leaf sizes 1 to 50, but with 20 bags for BagLearner.
    Data unshuffled.
    -----------------------------------------------------------------------------------'''
    baglearner_insample_rmse = []
    baglearner_outsample_rmse = []

    for i in range(1, 51):

        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)

        learner.addEvidence(trainX, trainY)  # train it
        #print learner.author()

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        baglearner_insample_rmse.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        baglearner_outsample_rmse.append(rmse)

    x = np.arange(1, 51)
    fig, ax = plt.subplots()

    ax.plot(x, baglearner_insample_rmse, label="In sample RMSE")
    ax.plot(x, baglearner_outsample_rmse, label="Out of Sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
           title="RMSE vs. Leaf Size for Bag Learner \n "
                 "Using Decision Tree Learner with Unshuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    #plt.show()
    fig.savefig('BL_unshuffled.png')
    plt.close()

    '''
     Test whether bagging can reduce or eliminate overfitting with respect to leaf size -
     Use DTlearner with leaf sizes 1 to 50, but with 20 bags for BagLearner.
     Shuffled data.
     -----------------------------------------------------------------------------------'''
    bl_shuffled_insample_rmse = []
    bl_shuffled_outsample_rmse = []

    for i in range(1, 51):

        # Shuffle data
        np.random.shuffle(trainX)
        np.random.shuffle(trainY)
        np.random.shuffle(testX)
        np.random.shuffle(testY)

        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)

        learner.addEvidence(trainX, trainY)  # train it
        #print learner.author()

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        bl_shuffled_insample_rmse.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        bl_shuffled_outsample_rmse.append(rmse)

    x = np.arange(1, 51)
    fig, ax = plt.subplots()

    ax.plot(x, bl_shuffled_insample_rmse, label="In sample RMSE")
    ax.plot(x, bl_shuffled_outsample_rmse, label="Out of Sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
           title="RMSE vs. Leaf Size for Bag Learner \n "
                 "Using Decision Tree Learner with Shuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)

    # plt.show()
    fig.savefig('BL_Shuffled.png')
    plt.close()
    '''----------------------------------------------------------------------------------------'''

    '''
    Part 3: Compare DT and RT Learners - Unshuffled Data
    ------------------------------------------------------------------------------------------'''
    DT_insample_rmse = []
    DT_outsample_rmse = []
    RT_insample_rmse = []
    RT_outsample_rmse = []

    for i in range(1, 51):
        learner = dt.DTLearner(leaf_size=i, verbose=False)

        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        DT_insample_rmse.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        DT_outsample_rmse.append(rmse)

    for i in range(1, 51):
        learner = rt.RTLearner(leaf_size=i, verbose=False)

        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        RT_insample_rmse.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        RT_outsample_rmse.append(rmse)

    # Plot for in sample RMSE
    x = np.arange(1, 51)
    fig, ax = plt.subplots()

    ax.plot(x, DT_insample_rmse, label="DT in sample RMSE")
    ax.plot(x, RT_insample_rmse, label="RT in sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
            title="In Sample RMSE for Decision Tree and Random Tree Learners \n with Unshuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)

    fig.savefig('DT_vs_RT_insampleRMSE.png')
    plt.close()

    # Plot for outsample RMSE
    x = np.arange(1, 51)
    fig, ax = plt.subplots()

    ax.plot(x, DT_outsample_rmse, label="DT out sample RMSE")
    ax.plot(x, RT_outsample_rmse, label="RT out sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
            title="Out Sample RMSE for Decision Tree and Random Tree Learners \n with Unshuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.savefig('DT_vs_RT_outsampleRMSE.png')
    plt.close()
    '''-----------------------------------------------------------------------------------------------'''

    '''
    Part 3: Compare DT and RT Learners - Shuffled Data
    ------------------------------------------------------------------------------------------'''
    DT_insample_rmse = []
    DT_outsample_rmse = []
    RT_insample_rmse = []
    RT_outsample_rmse = []

    for i in range(1, 51):

        # Shuffle data
        np.random.shuffle(trainX)
        np.random.shuffle(trainY)
        np.random.shuffle(testX)
        np.random.shuffle(testY)

        learner = dt.DTLearner(leaf_size=i, verbose=False)

        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        DT_insample_rmse.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        DT_outsample_rmse.append(rmse)

    for i in range(1, 51):

        # Shuffle data
        np.random.shuffle(trainX)
        np.random.shuffle(trainY)
        np.random.shuffle(testX)
        np.random.shuffle(testY)

        learner = rt.RTLearner(leaf_size=i, verbose=False)

        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        RT_insample_rmse.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        RT_outsample_rmse.append(rmse)

    # Plot for in sample RMSE
    x = np.arange(1, 51)
    fig, ax = plt.subplots()

    ax.plot(x, DT_insample_rmse, label="DT in sample RMSE")
    ax.plot(x, RT_insample_rmse, label="RT in sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
           title="In Sample RMSE for Decision Tree and Random Tree Learners \n with Shuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)

    fig.savefig('DT_vs_RT_insampleRMSE_Shuffled.png')
    plt.close()

    # Plot for outsample RMSE
    x = np.arange(1, 51)
    fig, ax = plt.subplots()

    ax.plot(x, DT_outsample_rmse, label="DT out sample RMSE")
    ax.plot(x, RT_outsample_rmse, label="RT out sample RMSE")

    ax.set_xlim(left=1, right=51)

    ax.set(xlabel="Leaf Size", ylabel="Root Mean Square Error (RMSE)",
           title="Out Sample RMSE for Decision Tree and Random Tree Learners \n with Shuffled Data")
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.savefig('DT_vs_RT_outsampleRMSE_Shuffled.png')
    plt.close()


