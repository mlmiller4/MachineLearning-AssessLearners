import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt


class InsaneLearner(object):

    def __init__(self, verbose=False):
        #pass  # move along, these aren't the drones you're looking for

        self.learners = []  # List of learners

        # Load different learners into the learnerList array
        bagNum=20

        for i in range(0, bagNum):
            self.learners.append(bl.BagLearner(lrl.LinRegLearner, kwargs={}, bags=bagNum))


    def author(self):
        return 'mmiller319'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
         @summary: Add training data to learner
         @param dataX: X values of data to add
         @param dataY: the Y training values
         """
        # Add dataX and dataY to the .addEvidence() method for each learner
        for currentLearner in self.learners:
            currentLearner.addEvidence(dataX, dataY)

    # Query results of each of the LRL learners in the bags and return mean value
    def query(self,points):

        queryResults = []

        for currentLearner in self.learners:
            queryResults.append(currentLearner.query(points))

        queryResults = np.array(queryResults)

        return np.mean(queryResults, axis=0)


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"