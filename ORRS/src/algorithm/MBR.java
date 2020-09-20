/**
 * @author Fan Min (minfanphd@163.com), Mei Zheng and Heng-Ru Zhang 
 */
package algorithm;

import java.io.*;


public class MBR {
	/**
	 * The total number of numUsers
	 */
	private int numUsers;

	/**
	 * The total number of items
	 */
	private int numItems;

	/**
	 * The total number of ratings (non-zero values)
	 */
	private int numRatings;

	/**
	 * The average rating of all users on all items.
	 */
	private double overallAverageRating;

	/**
	 * The predictions.
	 */
	private double[] mbrPredictions;

	/**
	 * Compressed rating matrix. User-item-rating triples.
	 */
	private double[][] compressedRatingMatrix;

	/**
	 * The degree of users (how many item he has rated).
	 */
	private int[] userDegrees;

	/**
	 * The average rating of the current user.
	 */
	private double[] userAverageRatings;

	/**
	 * The degree of users (how many item he has rated).
	 */
	private int[] itemDegrees;

	/**
	 * The average rating of the current item.
	 */
	private double[] itemAverageRatings;

	/**
	 * The first user start from 0. Let the first user has x ratings, the second
	 * user will start from x.
	 */
	private int[] userStartingIndices;

	/**
	 * Number of non-neighbor objects.
	 */
	private int numNonNeighbors;

	/**
	 * The radius (delta) for determining the neighborhood.
	 */
	private double radius;

	/**
	 ************************* 
	 * Construct the rating matrix.
	 * 
	 * @param paraRatingFilename
	 *            the rating filename.
	 * @param paraNumUsers
	 *            number of users
	 * @param paraNumItems
	 *            number of items
	 * @param paraNumRatings
	 *            number of ratings
	 ************************* 
	 */
	public MBR(double[][] paraRatingMatrix, int paraNumUsers, int paraNumItems, int paraNumRatings) throws Exception {
		// Initialize these arrays
		numItems = paraNumItems;
		numUsers = paraNumUsers;
		numRatings = paraNumRatings;
		overallAverageRating = 0;

		userDegrees = new int[numUsers];
		userStartingIndices = new int[numUsers + 1];
		userAverageRatings = new double[numUsers];
		itemDegrees = new int[numItems];
		compressedRatingMatrix = new double[numRatings][3];
		itemAverageRatings = new double[numItems];

		mbrPredictions = new double[numRatings];

		int tempIndex = 0;
		userStartingIndices[0] = 0;
		userStartingIndices[numUsers] = numRatings;
		for (int i = 0; i < paraRatingMatrix.length; i++) {// USER
			for (int j = 0; j < paraRatingMatrix[0].length; j++) {// ITEM
				if (paraRatingMatrix[i][j] > 1e-6) {
					compressedRatingMatrix[tempIndex][0] = i;
					compressedRatingMatrix[tempIndex][1] = j;
					compressedRatingMatrix[tempIndex][2] = paraRatingMatrix[i][j];
					overallAverageRating += compressedRatingMatrix[tempIndex][2];
					userDegrees[(int) compressedRatingMatrix[tempIndex][0]]++;// [.]:userid，the times of user rating
					itemDegrees[(int) compressedRatingMatrix[tempIndex][1]]++;// [.]:itemid, the times of item rated

					if (tempIndex > 0) {
						// Starting to read the data of a new user.
						if (compressedRatingMatrix[tempIndex][0] != compressedRatingMatrix[tempIndex - 1][0]) {
							userStartingIndices[(int) compressedRatingMatrix[tempIndex][0]] = tempIndex;
						} // Of if
					} // Of if
					tempIndex++;
				} // Of if
			} // Of for j
		} // Of for i

		double[] tempUserTotalScore = new double[numUsers];
		double[] tempItemTotalScore = new double[numItems];
		for (int i = 0; i < numRatings; i++) {
			tempUserTotalScore[(int) compressedRatingMatrix[i][0]] += compressedRatingMatrix[i][2];
			tempItemTotalScore[(int) compressedRatingMatrix[i][1]] += compressedRatingMatrix[i][2];
		} // Of for i

		for (int i = 0; i < numUsers; i++) {
			userAverageRatings[i] = tempUserTotalScore[i] / userDegrees[i];
		} // Of for i
		for (int i = 0; i < numItems; i++) {
			itemAverageRatings[i] = tempItemTotalScore[i] / itemDegrees[i];
		} // Of for i
		overallAverageRating /= numRatings;
	}// Of the first constructor

	/**
	 ************************* 
	 * Set the radius (delta).
	 * 
	 * @author Fan Min
	 ************************* 
	 */
	public void setRadius(double paraRadius) {
		if (paraRadius > 0) {
			radius = paraRadius;
		} else {
			radius = 0.00001;
		} // Of if
	}// of setRadius

	/**
	 ************************* 
	 * Leave one out prediction. The predicted values are stored in mbrPredictions.
	 * 
	 * @see mbrPredictions
	 * @author Fan Min
	 ************************* 
	 */
	public void leaveOneOutPrediction() {
		double tempItemAverageRating;
		int tempUser, tempItem, tempRating;// Make each line of the code
											// shorter.
		System.out.println("\r\nLeaveOneOutPrediction for radius " + radius);

		numNonNeighbors = 0;
		for (int i = 0; i < numRatings; i++) {
			tempUser = (int) compressedRatingMatrix[i][0];
			tempItem = (int) compressedRatingMatrix[i][1];
			tempRating = (int) compressedRatingMatrix[i][2];

			// Step 1. Recompute average rating of the current item.
			tempItemAverageRating = (itemAverageRatings[tempItem] * itemDegrees[tempItem] - tempRating)
					/ (itemDegrees[tempItem] - 1);

			// Step 2. Recompute neighbors, at the same time obtain the ratings
			// of neighbors.
			int tempNeighbors = 0;
			double tempTotal = 0;
			int tempComparedItem;
			for (int j = userStartingIndices[tempUser]; j < userStartingIndices[tempUser + 1]; j++) {
				if (tempItem == (int) compressedRatingMatrix[j][1]) {
					continue;// Ignore itself.
				} // Of if

				tempComparedItem = (int) compressedRatingMatrix[j][1];
				if (Math.abs(tempItemAverageRating - itemAverageRatings[tempComparedItem]) < radius) {
					tempTotal += compressedRatingMatrix[j][2];
					tempNeighbors++;
				} // Of if
			} // Of for j

			if (tempNeighbors > 0) {
				mbrPredictions[i] = tempTotal / tempNeighbors;
			} else {
				mbrPredictions[i] = overallAverageRating;
				numNonNeighbors++;
			} // Of if
		} // Of for i
	}// Of leaveOneOutPrediction

	/**
	 ************************* 
	 * Compute the MAE based on the deviation of each leave-one-out.
	 * 
	 * @author Fan Min
	 ************************* 
	 */
	public double computeMAE() throws Exception {
		double tempTotalError = 0;
		for (int i = 0; i < mbrPredictions.length; i++) {
			tempTotalError += Math.abs(mbrPredictions[i] - compressedRatingMatrix[i][2]);
		} // Of for i
		System.out.println("Totoal error = " + tempTotalError);

		return tempTotalError / mbrPredictions.length;
	}// Of computeMAE

	/**
	 ************************* 
	 * Compute the MAE based on the deviation of each leave-one-out.
	 * 
	 * @author Fan Min
	 ************************* 
	 */
	public double computeRSME() throws Exception {
		double tempTotalError = 0;
		for (int i = 0; i < mbrPredictions.length; i++) {
			tempTotalError += (mbrPredictions[i] - compressedRatingMatrix[i][2])
					* (mbrPredictions[i] - compressedRatingMatrix[i][2]);
		} // Of for i

		double tempAverage = tempTotalError / mbrPredictions.length;

		return Math.sqrt(tempAverage);
	}// Of computeRSME

	public double[] getPrediction() {
		return mbrPredictions;
	}
	/**
	 ************************* 
	 * The main entrance
	 * 
	 * @author Fan Min
	 ************************* 
	 */
	 

}// Of class MBR
