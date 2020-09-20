package datamodel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

//import tool.DistrFun;
import tool.SimpleTool;

/**
 * Store and process the data. Split into training and testing sets, etc.
 * 
 * @author Yuan-yuan Xu, Fan Min
 *
 */
public class DataInfor {

	/**
	 * The number of users.
	 */
	int numUsers;

	/**
	 * The number of items.
	 */
	int numItems;

	/**
	 * The number of non-zero elements in training set.
	 */
	// public int numTrainingNonZero;

	/**
	 * The number of non-zero elements in testing set.
	 */
	int numTestingSize;

	/**
	 * Total number of ratings in the rating matrix.
	 */
	int totalNumRatings;

	/**
	 * Number of Gaussian noise.
	 */
	int numNoise;

	/********************** Feature Matrix ***********************************/
	/**
	 * The rank of small matrices. It is often 2 or 4. It is also the number of
	 * latent variable.
	 */
	int rank;

	/**
	 * The original rating matrix. It is uncompressed. 0 indicates not rated.
	 */
	double[][] ratingMatrix;

	/**
	 * The matrix of training set.
	 */
	public double[][] trainMatrix;

	/**
	 * The matrix of testing set.
	 */
	public double[][] testMatrix;

	/**
	 * Indicate which position (row, column) has non-zero values. The elements
	 * usually take values 0 or 1.
	 */
	public double[][] nonZeroIndexMatrix;
	public double[][] nonZeroTestIndexMatrix;
	public double[][] nonZeroTrainIndexMatrix;

	/**
	 * The noise type of each rating. [0.2, 0.1, 0.7] indicates that the
	 * probability is 0.2 for the first noise.
	 */
	double[][] initialRatingsNoiseDistribution;

	/**
	 * The weight of every noise.
	 */
	public double[] noiseWeight;

	/**
	 * The absolute position of testing data.
	 */
	private int[] positionOfTestingData;
	private int[] positionOfTrainingData;
	/**
	 * The mean/average value of rating for the training set.
	 * It is equal to sum(trainVector)/trainVector.length
	 */
	public double meanRatingOfTrain;

	/**
	 * Store non-zero element of training set as a vector.
	 * It is a compressed version with out considering the position.
	 */
	public double[] trainVector;
	// public double medianofTrain;

	/**
	 * The small matrix U.
	 */
	public double[][] subU;

	/**
	 * The small matrix V. U * V approximate R (non-zero part).
	 */
	public double[][] subV;

	public static final String SPLIT_SIGN = new String("	");

	/**
	 ********************** 
	 * Read the dataset
	 * 
	 * @param paraFilename
	 *            The file storing the data.
	 * @param paraNumUsers
	 *            Number of users. This might be obtained through scanning the
	 *            whole data. However, the approach requires an additional
	 *            scanning. Therefore we do not adopt it.
	 * @param paraNumItems
	 *            Number of items.
	 * @throws IOException
	 *             It may occur if the given file does not exist.
	 ********************** 
	 */
	public DataInfor(String paraFilename, int paraNumUsers, int paraNumItems)
			throws IOException {
		numUsers = paraNumUsers;
		numItems = paraNumItems;

		// Initialize matrices
		ratingMatrix = new double[numUsers][numItems];
		nonZeroIndexMatrix = new double[numUsers][numItems];
		trainMatrix = new double[numUsers][numItems];
		nonZeroTrainIndexMatrix = new double[numUsers][numItems];
		testMatrix = new double[numUsers][numItems];
		nonZeroTestIndexMatrix = new double[numUsers][numItems];

		meanRatingOfTrain = 0;
		numTestingSize = 0;

		// Read the data
		int tempUserIndex, tempItemIndex;
		double tempRating;
		totalNumRatings = 0;
		File file = new File(paraFilename);
		BufferedReader buffRead = new BufferedReader(new InputStreamReader(
				new FileInputStream(file)));
		while (buffRead.ready()) {
			String str = buffRead.readLine();
			String[] parts = str.split(SPLIT_SIGN);
			tempUserIndex = Integer.parseInt(parts[0]) - 1;// user id
			tempItemIndex = Integer.parseInt(parts[1]) - 1;// item id
			tempRating = Double.parseDouble(parts[2]);// rating
			ratingMatrix[tempUserIndex][tempItemIndex] = tempRating;
			totalNumRatings++;
		} // Of while
        //SimpleTool.printMatrix(ratingMatrix);
		observeOriginalMatrix(ratingMatrix);
		buffRead.close();
	}// of DataInfor

	/**
	 ********************** 
	 * Get the number of users.
	 ********************** 
	 */
	public int getNumUsers() {
		return ratingMatrix.length;
	}// Of getNumUsers

	/**
	 ********************** 
	 * Get the number of items.
	 ********************** 
	 */
	public int getNumItems() {
		return ratingMatrix[0].length;
	}// Of getNumItems

	/**
	 ********************** 
	 * Get the row number of the testing data.
	 * 
	 * @param paraTestingDataIndex
	 *            The index of the testing data. A position in the testing
	 *            array.
	 ********************** 
	 */
	public int getTestingDataRow(int paraTestingDataIndex) {
		return positionOfTestingData[paraTestingDataIndex] / numItems;
	}// Of getTestingDataRow

	public int getTrainingDataRow(int paraTrainingDataIndex) {
		return positionOfTrainingData[paraTrainingDataIndex] / numItems;
	}// Of getTrainingDataRow
	/**
	 ********************** 
	 * Get the column number of the testing data.
	 * 
	 * @param paraTestingDataIndex
	 *            The index of the testing data. A position in the testing
	 *            array.
	 ********************** 
	 */
	public int getTestingDataColumn(int paraTestingDataIndex) {
		return positionOfTestingData[paraTestingDataIndex] % numItems;
	}// Of getTestingDataColumn

	public int getTrainingDataColumn(int paraTrainingDataIndex) {
		return positionOfTrainingData[paraTrainingDataIndex] % numItems;
	}// Of getTestingDataColumn
	/**
	 ********************** 
	 * Get the size of the training set, i.e., the number of non-zero ratings.
	 * 
	 * @return the size of the training set.
	 ********************** 
	 */
	public int getTrainingSize() {
		return totalNumRatings - numTestingSize;
	}// Of getTrainingSize

	/**
	 ********************** 
	 * Get the size of the testing set, i.e., the number of non-zero ratings.
	 * 
	 * @return the size of the testing set.
	 ********************** 
	 */
	public int getTestingSize() {
		System.out.printf("numTestingSize: %d\r\n", numTestingSize);
		return numTestingSize;
	}// Of getTestingSize

	/**
	 ********************** 
	 * Get the rank of the factorization matrices. It is often 2 or 4.
	 * 
	 * @return the rank.
	 ********************** 
	 */
	public int getLowRank() {
		return rank;
	}//Of getLowRank

	/**
	 ********************** 
	 * Get the number of noises. It is often 3.
	 * 
	 * @return the rank.
	 ********************** 
	 */
	public int getNumNoise() {
		return numNoise;
	}//of getNumNoise

	public double[][] getNoiseDistribution(){
		return initialRatingsNoiseDistribution;
	}
	
	/**
	 ********************** 
	 * Split the data to obtain the training and testing sets, respectively. At
	 * the same time, obtain numTrainingNonZero and numTestingNonZero.
	 * 
	 * @param paraTestProportion
	 *            the proportion of the testing set.
	 ********************** 
	 */
	public void splitTrainAndTest(double paraTestProportion, int paraThresholdForEachColumn) {
		int[] tempIndxOfNonZero = new int[ratingMatrix.length];
		double tempTestingSizeForCurrentItem = 0;
		int[] tempRandPerm;
		int tempThresholdForEachColumn = paraThresholdForEachColumn;
		int tempPositionIndex = 0;
		
		positionOfTrainingData = new int[(int) (totalNumRatings * (1 -
				paraTestProportion))];
		positionOfTestingData = new int[(int) (totalNumRatings * paraTestProportion)];

		for (int j = 0; j < ratingMatrix[0].length; j++) {
			int tempCnt = 0;
			for (int i = 0; i < ratingMatrix.length; i++) {
				if (ratingMatrix[i][j] >= 1) {
					nonZeroIndexMatrix[i][j] = 1;
					tempIndxOfNonZero[tempCnt] = i;
					positionOfTrainingData[tempCnt] = i * numItems +j;
					tempCnt++;
				} // of if
			} // of for i
			// System.out.println("tempCnt:" + tempCnt);
			// System.out.println("j:" + j);
			tempRandPerm = new int[tempCnt];
			tempRandPerm = SimpleTool.generateRandomSequence(tempCnt);

			// Avoid generating all-zero column for the training set
			if (tempCnt > tempThresholdForEachColumn) {
				tempTestingSizeForCurrentItem = Math.floor(tempCnt
						* paraTestProportion);
			} else if (tempCnt <= tempThresholdForEachColumn && tempCnt > 1 && paraTestProportion > 1e-6) {
				tempTestingSizeForCurrentItem = 1;
			} else if (tempCnt == 1) {
				tempTestingSizeForCurrentItem = 0;
			} // of if
			numTestingSize += tempTestingSizeForCurrentItem;

			if (tempTestingSizeForCurrentItem >= 1) {
				for (int k = 0; k < tempTestingSizeForCurrentItem; k++) {
					int tempRow = tempIndxOfNonZero[tempRandPerm[k]];
					testMatrix[tempRow][j] = ratingMatrix[tempRow][j];
					nonZeroTestIndexMatrix[tempRow][j] = 1;
					positionOfTestingData[tempPositionIndex] = tempRow
							* numItems + j;
					tempPositionIndex++;
				} // of for k
			} // of if
		} // of for j
//		  SimpleTool.printMatrix(nonZeroIndexMatrix);
		nonZeroTrainIndexMatrix = tool.MatrixOpr.Matrix_Sub(nonZeroIndexMatrix,
				nonZeroTestIndexMatrix);
		// System.out.println("numTrainingNonZero:" + numTrainingNonZero + "," +
		// "NumofTestNonzero:" + NumofTestNonzero);
		trainMatrix = tool.MatrixOpr.Matrix_Sub(ratingMatrix, testMatrix);
	}// of splitTrainAndTest

	/**
	 ********************** 
	 * Convert the training set into a vector. The result is stored in
	 * trainVector.
	 * 
	 * @see #generateRandomSubMatrix(int)
	 * @see tool.MatrixOpr#getMedian(double[])
	 ********************** 
	 */
	public void computeTrainVector() {
		trainVector = new double[getTrainingSize()];
		int tempCnt = 0;
		for (int i = 0; i < trainMatrix.length; i++) {
			for (int j = 0; j < trainMatrix[0].length; j++) {
				if (trainMatrix[i][j] > 1e-6) {
					trainVector[tempCnt] = trainMatrix[i][j];
					tempCnt++;
				}// of if
			}// of for j
		}// of for i
	}// of getTrainVector
	public double[][] getRatingMatrix(){
		return ratingMatrix;
	}//Of getRatingMatrix
	/**
	 ********************** 
	 * Compute the average rating of the training set.
	 ********************** 
	 */
	public void computeTrainingSetAverageRating() {
		double tempSum = tool.MatrixOpr.Matrix_Sum(trainMatrix);
		meanRatingOfTrain = tempSum / getTrainingSize();
	}// Of computeTrainingSetAverageRating

	/**
	 ********************** 
	 * Recompute the training set. Each rating subtracts the mean value.
	 * In this way the average value would be 0.
	 ********************** 
	 */
	public void recomputeTrainset() {
		for (int i = 0; i < trainMatrix.length; i++) {
			for (int j = 0; j < trainMatrix[0].length; j++) {
				if (nonZeroTrainIndexMatrix[i][j] > 1e-6) {
					trainMatrix[i][j] -= meanRatingOfTrain;
				} // of if
			} // of for j
		} // of for i
	}// Of recomputeTrainset

	/**
	 ********************** 
	 * Set the distribution of the noise on each rating
	 ********************** 
	 */
	public void setRandomNoiseDistribution(int paraNumNoise) {
		numNoise = paraNumNoise;
		Random tempRandom = new Random();
		initialRatingsNoiseDistribution = new double[getTrainingSize()][numNoise];
		for (int i = 0; i < getTrainingSize(); i++) {
			initialRatingsNoiseDistribution[i][tempRandom.nextInt(numNoise)] = 1;
		} // of for i
		// SimpleTool.printMatrix(initialRatingsNoiseDistribution);
	}// Of setRandomNoiseDistribution

	/**
	 ********************** 
	 * Compute the weight of each noise
	 ********************** 
	 */
	public void computeWeight() {
		noiseWeight = new double[numNoise];
		for (int j = 0; j < initialRatingsNoiseDistribution[0].length; j++) {
			double tempColSum = 0;
			for (int i = 0; i < initialRatingsNoiseDistribution.length; i++) {
				tempColSum += initialRatingsNoiseDistribution[i][j];
			} // of for i
			noiseWeight[j] = tempColSum / getTrainingSize();
		} // of for j
	}// of computeWeight

	/**
	 ********************** 
	 * Generate random sub matrices for initialization.
	 * The elements are subject to the uniform distribution in (-tempMu, tempMu).
	 ********************** 
	 */
	public void generateRandomSubMatrix(int paraRank) {
		rank = paraRank;
		subU = new double[numUsers][rank];
		subV = new double[numItems][rank];

		double tempMedianOfTrain = tool.MatrixOpr.getMedian(trainVector);
		double tempMu = Math.sqrt(tempMedianOfTrain / rank);
		// Step 1. Generate two gaussian sub-matrices
		for (int j = 0; j < rank; j++) {
			for (int i = 0; i < numUsers; i++) {
				subU[i][j] = Math.random() * 2 * tempMu - tempMu;
			} // of for i

			for (int i = 0; i < numItems; i++) {
				subV[i][j] = Math.random() * 2 * tempMu - tempMu;
			} // of for i
		} // of for j
		//SimpleTool.printMatrix(subU);
	}// of generateRandomSubMatrix

	/**
	 ********************** 
	 * Add the mean rating back so that we can compare the prediction with the actual one.
	 ********************** 
	 */
	public void recoverTrainMatrix() {
		for (int i = 0; i < trainMatrix.length; i++) {
			for (int j = 0; j < trainMatrix[0].length; j++) {
				if (nonZeroTrainIndexMatrix[i][j] > 1e-6) {
					trainMatrix[i][j] = trainMatrix[i][j] + meanRatingOfTrain;
				} // of if
			} // of for j
		} // of for i
	}// of recoverTrainMatrix
	
	public void observeOriginalMatrix(double paraMatrix[][]) {
		double tempSum, tempMAE = 0, tempVariance = 0;
		double tempAve;
		double tempTotalNum = getTrainingSize() + getTestingSize();
		tempSum = tool.MatrixOpr.Matrix_Sum(paraMatrix);
		tempAve = tempSum / tempTotalNum;
		System.out.printf("The mean value of R (OriginalMatrix) is %8.4f\r\n",tempAve);
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[0].length; j++) {
				if(paraMatrix[i][j] > 1e-6) {
					tempMAE += Math.abs(paraMatrix[i][j] - tempAve);
					tempVariance += (paraMatrix[i][j] - tempAve) * (paraMatrix[i][j] - tempAve);
				}//Of if
			}//Of for j
		}//Of for i
		tempMAE /= tempTotalNum;
		tempVariance /= tempTotalNum;
		System.out.printf("The variance  of R (OriginalMatrix) is %8.4f\r\n",tempVariance);
		System.out.printf("The MAE of R (OriginalMatrix) between its mean value: %8.4f\r\n", tempMAE);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			// Step 1. Initialize the train and test data based on group
			// information
			String tempFilename = new String("data/ml-100k/u.data");

			DataInfor tempData = new DataInfor(tempFilename, 943, 1682);
			tempData.splitTrainAndTest(0.2, 5);
			tempData.computeTrainingSetAverageRating();
			tempData.recomputeTrainset();
			tempData.computeTrainVector();
			tempData.setRandomNoiseDistribution(3);
			tempData.computeWeight();
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// of main

}//Of class DataInfor
