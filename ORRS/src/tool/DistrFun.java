package tool;

import java.util.Random;

public class DistrFun {
	/**
	 * 
	 * @param paraMu
	 * @return
	 */
	public static double uniform(double paraRange) {
		// Random r = new Random();

		double tempUni = 2 * paraRange * Math.random() - paraRange;

		return tempUni;
	}// Of uniform

	/**
	 * 
	 * @param paraMu
	 * @param paraSigma
	 * @return
	 */
	public static double gaussian(double paraMu, double paraSigma) {
		Random r = new Random();
		double tempGau = r.nextGaussian();

		return paraSigma * tempGau + paraMu;
	}// of gaussian
}//of class DistrFun
