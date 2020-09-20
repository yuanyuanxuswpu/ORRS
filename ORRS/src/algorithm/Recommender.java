package algorithm;

import java.util.Arrays;

import datamodel.RatingSystem2DBoolean;

/**
 * User based three-way recommender. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: January 31, 2019.<br>
 *       Last modified: January 31, 2020.
 * @version 1.0
 */

public abstract class Recommender {
	/**
	 * The dataset.
	 */
	RatingSystem2DBoolean dataset;

	/**
	 * Number of users.
	 */
	protected int numUsers;

	/**
	 * Number of items.
	 */
	protected int numItems;

	/**
	 * Number of ratings.
	 */
	protected int numRatings;

	/**
	 ************************ 
	 * The first constructor.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 ************************ 
	 */
	public Recommender(RatingSystem2DBoolean paraDataset) {
		dataset = paraDataset;

		initializeData();
	}// Of the first constructor

	/**
	 ************************ 
	 * Initialize some variables.
	 ************************ 
	 */
	void initializeData() {
		numUsers = dataset.getNumUsers();
		numItems = dataset.getNumItems();
		numRatings = dataset.getNumRatings();
	}// Of initializeData

	/**
	 *********************************** 
	 * Show me.
	 *********************************** 
	 */
	public String toString() {
		String resultString = "UserBasedThreeWayRecommender on dataset:\r\n" + dataset;
		return resultString;
	}// Of toString
}// Of class UserBasedThreeWayRecommender
