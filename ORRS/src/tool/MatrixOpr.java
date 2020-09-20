package tool;

import java.util.*;

public class MatrixOpr {
	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	 */
	public static double[][] Matrix_Mult(double[][] paraU, double[][] paraV) {
		double[][] tempResultMatrix = new double[paraU.length][paraV.length];
		if (paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraV.length; j++) {
					for (int k = 0; k < paraU[0].length; k++) {
						tempResultMatrix[i][j] += paraU[i][k] * paraV[j][k];
					} // of for k
				} // of for j
			} // of for i
		} // of if

		return tempResultMatrix;
	}// of Matrix_Multiply

	public static double[][] ColVec_Multi_RowVec(double[] paraColVec, double[] paraRowVec) {
		double[][] tempResultMatrix = new double[paraColVec.length][paraRowVec.length];
		for (int i = 0; i < paraColVec.length; i++) {
			for (int j = 0; j < paraRowVec.length; j++) {
				tempResultMatrix[i][j] = paraColVec[i] * paraRowVec[j];
			} // of for j
		} // of for i
		return tempResultMatrix;
	}// of Matrix_Multiply

	public static double Vector_Mult(double[] paraU, double[] paraV) {
		double tempResult = 0;
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResult += paraU[i] * paraV[i];
			} // of for i
		} // of if

		return tempResult;
	}// of dotMultiply

	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	 */
	public static double[][] Matrix_DotMult(double[][] paraU, double[][] paraV) {
		double[][] tempResultMatrix = new double[paraU.length][paraU[0].length];

		if (paraU.length == paraV.length && paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraU[0].length; j++) {
					tempResultMatrix[i][j] = paraU[i][j] * paraV[i][j];
					if (Math.abs(tempResultMatrix[i][j]) == 0.0 // To check
																// whether
																// tempResultMatrix[i][j]
																// is equal to
																// -0.0
							&& Math.copySign(1.0, tempResultMatrix[i][j]) < 0.0) {
						tempResultMatrix[i][j] = 0.0;
					} // of if
				} // of for j
			} // of for i
		} // of if
		return tempResultMatrix;
	}// of Matrix_DotMult

	public static double[] Vector_DotMult(double[] paraU, double[] paraV) {
		double[] tempResultVector = new double[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResultVector[i] = paraU[i] * paraV[i];
			} // of for i
		} // of if
		return tempResultVector;
	}// of Vector_DotMult

	public static double[][] matrixDotDivision(double[][] paraU, double[][] paraV) {
		double[][] resultVector = new double[paraU.length][];
	
		for (int i = 0; i < paraU.length; i++) {
			resultVector[i] = new double[paraU[i].length];
			for (int j = 0; j < paraU[i].length; j++) {
				if (paraV[i][j] != 0) {
					resultVector[i][j] = paraU[i][j] / paraV[i][j];
				} // Of if
			} // Of for j
		} // of for i

		return resultVector;
	}// of Vector_DotDiv

	public static double[] Vector_DotDiv(double[] paraU, double[] paraV) {
		double[] tempResultVector = new double[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				if (paraV[i] != 0) {
					tempResultVector[i] = paraU[i] / paraV[i];
				} // of if
			} // of for i
		} // of if
		return tempResultVector;
	}// of Vector_DotDiv

	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	 */
	public static double[][] Matrix_Add(double[][] paraU, double[][] paraV) {
		double[][] tempResultMatrix = new double[paraU.length][paraU[0].length];
		if (paraU.length == paraV.length && paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraU[0].length; j++) {
					tempResultMatrix[i][j] = paraU[i][j] + paraV[i][j];
				} // of for j
			} // of for i
		} // Of if
		return tempResultMatrix;
	}// of add

	public static double[][] Matrix_Sub(double[][] paraU, double[][] paraV) {
		double[][] tempResultMatrix = new double[paraU.length][paraV[0].length];
		if (paraU.length == paraV.length && paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraU[0].length; j++) {
					tempResultMatrix[i][j] = paraU[i][j] - paraV[i][j];
				} // of for j
			} // of for i
		} // of if
		return tempResultMatrix;
	}// of Matrix_Sub

	public static double[] Vector_Sub(double[] paraU, double[] paraV) {
		double[] tempResultVector = new double[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResultVector[i] = paraU[i] - paraV[i];
			} // of for i
		} // of if

		return tempResultVector;
	}// of Vector_Sub

	public static double[] Vector_Add(double[] paraU, double[] paraV) {
		double[] tempResultVector = new double[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResultVector[i] = paraU[i] + paraV[i];
			} // of for i
		} // of if
		return tempResultVector;
	}// of Vector_Sub

	/*
	 * paraVector is considered as a column vector, and is expanded to a matrix.
	 * Each column of the matrix is paraVector. And the column number of the matrix
	 * is paraColNum.
	 */
	public static double[][] Vector2Matrix(double[] paraVector, int paraColNum) {
		double[][] tempResultMatrix = new double[paraVector.length][paraColNum];
		for (int j = 0; j < paraColNum; j++) {
			for (int i = 0; i < paraVector.length; i++) {
				tempResultMatrix[i][j] = paraVector[i];
			} // of for i
		} // of for j
		return tempResultMatrix;
	}// of Vector2Matrix

	public static double[] SumbyCol(double[][] paraMatrix) {
		double[] tempResult = new double[paraMatrix[0].length];
		for (int j = 0; j < paraMatrix[0].length; j++) {
			for (int i = 0; i < paraMatrix.length; i++) {
				tempResult[j] += paraMatrix[i][j];
			} // of for i
		} // of for j
		return tempResult;
	}// of SumCol

	public static double Matrix_Sum(double[][] paraMatrix) {
		double tempResult = 0;

		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[0].length; j++) {
				tempResult += paraMatrix[i][j];
			} // of for j
		} // of for i

		return tempResult;
	}// of Matrix_Sum

	public static double[][] Matrix_Transpose(double[][] paraMatrix) {
		double[][] tempResult = new double[paraMatrix[0].length][paraMatrix.length];

		for (int i = 0; i < tempResult.length; i++) {
			for (int j = 0; j < tempResult[0].length; j++) {
				tempResult[i][j] = paraMatrix[j][i];
			} // of for j
		} // of for i

		return tempResult;
	}// of Matrix_Transpose

	public static double[][] Add_MatrixandNumber(double[][] paraMatrix, double paraNum) {
		double[][] tempResult = new double[paraMatrix.length][paraMatrix[0].length];

		for (int i = 0; i < tempResult.length; i++) {
			for (int j = 0; j < tempResult[0].length; j++) {
				tempResult[i][j] = paraMatrix[i][j] + paraNum;
			} // of for j
		} // of for i

		return tempResult;
	}// of Add_MatrixandNumber

	public static double[][] Matrix_Subspace(double[][] paraMatrix1, double[][] paraMatrix2) {
		double[][] tempResult;
		double[][] tempMulti;
		// double tempMax = 0;
		// if (paraMatrix1.length == paraMatrix2.length && paraMatrix1[0].length
		// == paraMatrix2[0].length) {

		tempMulti = new double[paraMatrix1[0].length][paraMatrix1[0].length];
		tempResult = new double[tempMulti.length][tempMulti[0].length];

		tempMulti = Matrix_Mult(paraMatrix1, paraMatrix2);

		for (int i = 0; i < tempMulti.length; i++) {
			for (int j = 0; j < tempMulti[0].length; j++) {
				tempMulti[i][j] = Math.acos(Math.abs(tempMulti[i][j]));
			} // of for j
		} // of for i

		tempResult = tempMulti;
		return tempResult;
		// }//of if

		// return tempMulti;
	}// of Matrix_Subspace

	public static double getMedian(double[] arr) {
		double[] tempArr = Arrays.copyOf(arr, arr.length);
		Arrays.sort(tempArr);
		if (tempArr.length % 2 == 0) {
			return (tempArr[tempArr.length >> 1] + tempArr[(tempArr.length >> 1) - 1]) / 2;
		} else {
			return tempArr[(tempArr.length >> 1)];
		}
	}// of getMedian

	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	 */
	// public static double[][] Matrix_Col_Mult(double[][] paraU, double[][] paraV,
	// int paraFirstCol, int paraSecond) {
	// for (int i = 0; i < paraU.length; i++) {
	// paraU[i][paraFirstCol]
	// }
	// return null;
	// }
	// }

}// of MatrixOpr
