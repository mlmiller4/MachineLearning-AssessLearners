import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        # pass # move along, these aren't the drones you're looking for
        self.tree = None
        self.leaf_size = leaf_size

    def author(self):
        return 'mmiller319'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # Format dataY as an ndarray
        dataY = np.array([dataY])

        # Transpose dataY so it can be appended onto dataX
        dataY_transpose = dataY.T
        data_appended = np.append(dataX, dataY_transpose, axis=1)

        # Pass appended data to the build_tree function
        self.tree = self.build_tree(data_appended)


    # Implements the Decision Tree Algorithm of JR Quinlan
    def build_tree(self, data):

        # Base Case - when the number of nodes is < leaf_size, the stopping condition has been reached.
        if data.shape[0] < self.leaf_size:
            return np.array([['Leaf', np.mean(data[:, -1]), None, None]])

        # Check if all values of data.y are the same - if so return a leaf
        # .unique() checks how many unique values are in data - if its size =1, then all values are the same
        dataY = data[:, -1]
        if np.unique(dataY.size == 1):
            return np.array([['Leaf', data[0, -1], None, None]])

        else:
            # Find the best feature to split on
            bestFeature = int(self.featureToSplit(data))

            # The split value will be the median of the values for the best feature
            SplitVal = np.median(data[:, bestFeature])

            # If the split value is equal to the maximum value for the best feature, then no right
            # subtree will be formed.
            if SplitVal == max(data[:, bestFeature]):
                return np.array([['Leaf', np.mean(data[:, -1]), None, None]])

            # Otherwise, create a left and right subtree based on whether the feature values are
            # greater than or less than the split value
            leftSubTree = self.build_tree(data[data[:, bestFeature] <= SplitVal])
            rightSubTree = self.build_tree(data[data[:, bestFeature] > SplitVal])

            # Create an np.array as the root, append left and right subtrees to it and return the full tree
            root = np.array([[bestFeature, SplitVal, 1, leftSubTree.shape[0] + 1]])
            halfTree = np.append(root, leftSubTree, axis=0)
            fullTree = np.append(halfTree, rightSubTree, axis=0)
            return fullTree


    # Returns index of column which contains the best feature to base split on - this is the feature
    # with the highest correlation with the Y data
    def featureToSplit(self, data):

        # Separate the data back into dataX and dataY
        dataX_cols = data.shape[1]-1
        dataX = data[:, 0:dataX_cols]
        dataY = data[:, dataX_cols]

        #numCols = data.shape[1]
        #dataX = np.delete(data, numCols-1, axis=1)

        # Calculate correlations between each feature (SP, DAX, FTSE, etc.) and the corresponding label (EM data)
        # Uses numpy.corrcoef which returns the Pearson product-moment correlation coefficients
        correlations = []

        for i in range(0,dataX_cols):
            correlation = np.corrcoef(data[:,i], dataY)
            correlation = correlation[0,1]
            correlations.append(correlation)

        # Return the index of the feature with the highest correlation
        return np.argmax(correlations)


    def query(self, points):

        results = []    # Hold results

        # for each row, predict the value of the target (EM)
        for i in range(0, points.shape[0]):

            currentRow = points[i,:]    # get current row from points array
            treeRow = 0     # current row in walking of the tree

            while (self.tree[treeRow, 0] != 'Leaf'):
                currTreeRow = int(self.tree[treeRow, 0])
                SplitVal = self.tree[treeRow, 1]

                if currentRow[currTreeRow] <= SplitVal:
                    treeRow += int(self.tree[treeRow, 2])
                else:
                    treeRow += int(self.tree[treeRow, 3])


            # Get feature found in tree based on walking tree using the points[] array and append to results[]
            currentFeature = self.tree[treeRow, 1]
            results.append(currentFeature)

        return results



if __name__ == "__main__":
    print ""

