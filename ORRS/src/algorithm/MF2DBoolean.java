package algorithm;

import java.io.*;

import common.Common;
import datamodel.*;

/**
 * Implement two matrix factorization algorithms, where data is stand alone.
 * <br>
 * Project: Outlier removal recommender.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: December 3, 2019.<br>
 *       Last modified: September 10, 2020.
 * @version 1.0
 */

public class MF2DBoolean extends Recommender {

	/**
	 * A parameter for controlling learning regular.
	 */
	double alpha;

	/**
	 * A parameter for controlling the learning speed.
	 */
	double lambda;

	/**
	 * The low rank of the small matrices.
	 */
	int rank;

	/**
	 * The user matrix U.
	 */
	double[][] userSubspace;

	/**
	 * The item matrix V.
	 */
	double[][] itemSubspace;

	/**
	 * Regular scheme.
	 */
	int regularScheme;

	/**
	 * The range for the initial value of subspace values.
	 */
	double subspaceValueRange = 0.5;

	/**
	 * No regular scheme.
	 */
	public static final int NO_REGULAR = 0;

	/**
	 * No regular scheme.
	 */
	public static final int PQ_REGULAR = 1;

	/**
	 * How many rounds for training.
	 */
	int trainRounds = 200;

	/**
	 ************************ 
	 * The second constructor.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 ************************ 
	 */
	public MF2DBoolean(RatingSystem2DBoolean paraDataset) {
		super(paraDataset);
		//dataset = paraDataset;

		// Initialize some parameters.
		rank = 5;
		alpha = 0.0001;
		lambda = 0.005;
		regularScheme = 0;
	}// Of the second constructor

	/**
	 ************************ 
	 * Set parameters.
	 * 
	 * @param paraRank
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void setParameters(int paraRank, double paraAlpha, double paraLambda,
			int paraRegularScheme, int paraTrainRounds) {
		rank = paraRank;
		alpha = paraAlpha;
		lambda = paraLambda;
		regularScheme = paraRegularScheme;
		trainRounds = paraTrainRounds;
	}// Of setParameters

	/**
	 ************************ 
	 * Get parameters.
	 ************************ 
	 */
	public String getParameters() {
		String resultString = "" + rank + ", " + alpha + ", " + lambda + ", " + regularScheme + ", "
				+ trainRounds;
		return resultString;
	}// Of setParameters

	/**
	 ************************ 
	 * Initialize subspaces. Each value is in [-paraRange, +paraRange].
	 * 
	 * @paraRange The range of the initial values.
	 ************************ 
	 */
	void initializeSubspaces(double paraRange) {
		subspaceValueRange = paraRange;
		userSubspace = new double[dataset.getNumUsers()][rank];

		for (int i = 0; i < dataset.getNumUsers(); i++) {
			for (int j = 0; j < rank; j++) {
				userSubspace[i][j] = (Common.random.nextDouble() - 0.5) * 2 * subspaceValueRange;
			} // of for j
		} // Of for i

		// SimpleTool.printMatrix(DataInfo.userFeature);
		itemSubspace = new double[dataset.getNumItems()][rank];
		for (int i = 0; i < dataset.getNumItems(); i++) {
			for (int j = 0; j < rank; j++) {
				itemSubspace[i][j] = (Common.random.nextDouble() - 0.5) * 2 * subspaceValueRange;
			} // Of for j
		} // Of for i
	}// Of initializeSubspaces

	/**
	 ************************ 
	 * Predict the rating of the user to the item
	 * 
	 * @param paraUser
	 *            The user index.
	 ************************ 
	 */
	public double predict(int paraUser, int paraItem) {
		double resultValue = 0;
		for (int i = 0; i < rank; i++) {
			resultValue += userSubspace[paraUser][i] * itemSubspace[paraItem][i];
		} // Of for i
		return resultValue;
	}// Of predict

	/**
	 ************************ 
	 * Predict the ratings of the user to each item.
	 * 
	 * @param paraUser
	 *            The user index.
	 ************************ 
	public double[] predictForUser(int paraUser) {
		// System.out.println("predictForUser(" + paraUser + ")");
		double[] resultPredictions = new double[dataset.getNumItems()];
		for (int i = 0; i < dataset.getNumItems(); i++) {
			resultPredictions[i] = predict(paraUser, i);
		} // Of for i
		return resultPredictions;
	}// Of predictForUser
	 */

	/**
	 ************************ 
	 * Train.
	 * 
	 * @param paraRounds
	 *            The number of rounds.
	 ************************ 
	 */
	public void train() {
		train(trainRounds);
	}// Of train

	/**
	 ************************ 
	 * Train.
	 * 
	 * @param paraRounds
	 *            The number of rounds.
	 ************************ 
	 */
	public void train(int paraRounds) {
		for (int i = 0; i < paraRounds; i++) {
			update();
			if (i % 50 == 0) {
				// Show the process
				System.out.println("Round " + i);
				// System.out.println("MAE: " + mae());
			} // Of if
		} // Of for i
	}// Of train

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void update() {
		switch (regularScheme) {
		case NO_REGULAR:
			updateNoRegular();
			break;
		case PQ_REGULAR:
			updatePQRegular();
			break;
		default:
			System.out.println("Unsupported regular scheme: " + regularScheme);
			System.exit(0);
		}// Of switch
	}// Of update

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updateNoRegular() {
		for (int i = 0; i < dataset.getNumUsers(); i++) {
			for (int j = 0; j < dataset.getUserNumRatings(i); j++) {
				// Ignore the testing set.
				if (!dataset.getTrainIndication(i, j)) {
					continue;
				} // Of if

				Triple tempTriple = dataset.getTriple(i, j);
				int tempUserId = tempTriple.user;
				int tempItemId = tempTriple.item;
				double tempRate = tempTriple.rating;

				double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
				// tempResidual = Math.abs(tempResidual);

				// Update user subspace
				double tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k];
					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k];

					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of updateNoRegular

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updatePQRegular() {
		for (int i = 0; i < dataset.getNumUsers(); i++) {
			for (int j = 0; j < dataset.getUserNumRatings(i); j++) {
				// Ignore the testing set.
				if (!dataset.getTrainIndication(i, j)) {
					continue;
				} // Of if

				Triple tempTriple = dataset.getTriple(i, j);
				int tempUserId = tempTriple.user;
				int tempItemId = tempTriple.item;
				double tempRate = tempTriple.rating;

				double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
				// tempResidual = Math.abs(tempResidual);

				// Update user subspace
				double tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k]
							- lambda * userSubspace[tempUserId][k];

					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k]
							- lambda * itemSubspace[tempItemId][k];
					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of updatePQRegular

	/**
	 ************************ 
	 * Compute the RSME.
	 * 
	 * @return RSME of the current factorization.
	 ************************ 
	 */
	public double rsme() {
		double resultRsme = 0;
		int tempTestCount = 0;

		for (int i = 0; i < dataset.getNumUsers(); i++) {
			for (int j = 0; j < dataset.getUserNumRatings(i); j++) {
				// Ignore the training set.
				if (dataset.getTrainIndication(i, j)) {
					continue;
				} // Of if

				Triple tempTriple = dataset.getTriple(i, j);
				int tempUserId = tempTriple.user;
				int tempItemId = tempTriple.item;
				double tempRate = tempTriple.rating;

				double tempPrediction = predict(tempUserId, tempItemId);// +
																		// DataInfo.mean_rating;

				if (tempPrediction < dataset.getRatingLowerBound()) {
					tempPrediction = dataset.getRatingLowerBound();
				} else if (tempPrediction > dataset.getRatingUpperBound()) {
					tempPrediction = dataset.getRatingUpperBound();
				} // Of if

				double tempError = tempRate - tempPrediction;
				resultRsme += tempError * tempError;
				tempTestCount++;
			} // Of for j
		} // Of for i

		return Math.sqrt(resultRsme / tempTestCount);
	}// Of rsme

	/**
	 ************************ 
	 * Compute the MAE.
	 * 
	 * @return MAE of the current factorization.
	 ************************ 
	 */
	public double mae() {
		double resultMae = 0;
		int tempTestCount = 0;

		for (int i = 0; i < dataset.getNumUsers(); i++) {
			for (int j = 0; j < dataset.getUserNumRatings(i); j++) {
				// Ignore the training set.
				if (dataset.getTrainIndication(i, j)) {
					continue;
				} // Of if

				Triple tempTriple = dataset.getTriple(i, j);
				int tempUserId = tempTriple.user;
				int tempItemId = tempTriple.item;
				double tempRate = tempTriple.rating;

				double tempPrediction = predict(tempUserId, tempItemId);

				if (tempPrediction < dataset.getRatingLowerBound()) {
					tempPrediction = dataset.getRatingLowerBound();
				} else if (tempPrediction > dataset.getRatingUpperBound()) {
					tempPrediction = dataset.getRatingUpperBound();
				} // Of if

				double tempError = tempRate - tempPrediction;

				resultMae += Math.abs(tempError);
				// System.out.println("resultMae: " + resultMae);
				tempTestCount++;
			} // Of for j
		} // Of for i

		return (resultMae / tempTestCount);
	}// Of mae

	/**
	 ************************ 
	 * The training testing scenario.
	 ************************ 
	 */
	public static void testTrainingTesting(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			double paraLikeThreshold, boolean paraCompress, int paraRounds) {
		try {
			// Step 1. read the training and testing data
			RatingSystem2DBoolean tempDataset = new RatingSystem2DBoolean(paraFilename,
					paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound,
					paraRatingUpperBound, paraLikeThreshold, paraCompress);

			tempDataset.initializeTraining(0.8);

			MF2DBoolean tempLearner = new MF2DBoolean(tempDataset);

			tempLearner.setParameters(10, 0.0001, 0.005, PQ_REGULAR, paraRounds);
			// tempMF.setTestingSetRemainder(2);
			// Step 2. Initialize the feature matrices U and V
			tempLearner.initializeSubspaces(0.5);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");

			tempLearner.train();

			double tempMAE = tempLearner.mae();
			double tempRSME = tempLearner.rsme();
			System.out.println("Finally, MAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of testTrainingTesting

	/**
	 ************************ 
	 * Training and testing using the same data.
	 ************************ 
	 */
	public static void testAllTrainingTesting(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings, double paraRatingLowerBound,
			double paraRatingUpperBound, double paraLikeThreshold, boolean paraCompress,
			int paraRounds) {
		try {
			// Step 1. read the training and testing data
			RatingSystem2DBoolean tempDataset = new RatingSystem2DBoolean(paraFilename,
					paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound,
					paraRatingUpperBound, paraLikeThreshold, paraCompress);

			tempDataset.initializeTraining(1.0);

			MF2DBoolean tempLearner = new MF2DBoolean(tempDataset);
			// tempMF.setTestingSetRemainder(2);
			// Step 2. Initialize the feature matrices U and V
			
			tempLearner.setParameters(10, 0.0001, 0.005, NO_REGULAR, paraRounds);
			tempLearner.initializeSubspaces(0.5);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");

			tempLearner.train();

			tempDataset.initializeTraining(0);
			double tempMAE = tempLearner.mae();
			double tempRSME = tempLearner.rsme();
			System.out.println("Finally, MAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of testAllTrainingTesting

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		//testAllTrainingTesting("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10,
		//		0.5, false, 150);
		//testAllTrainingTesting("data/movielens943u1682m.txt", 943, 1682, 100000, 1, 5, 3.5, true, 350);
		// testSameTrainingTesting("data/jester-data-1/jester-data-1.txt",
		// 24983, 101, 1810455, -10, 10, 500);
		testTrainingTesting("data/movielens943u1682m.txt", 943, 1682, 100000, 1, 5, 3.5, true, 500);
	}// Of main
}// Of class MF2DBoolean
