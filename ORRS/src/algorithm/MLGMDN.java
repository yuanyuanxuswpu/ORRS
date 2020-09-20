/**
 * @author Yuan-Yuan Xu, Fan Min (minfanphd@163.com), Heng-Ru Zhang,

 * Reference: De-Yu Meng.... in matlab
 * Part I
 * Input: movielens-100k, 943 users and 1682 items (considering the ratings matrix R with noises of " Mixture of Gaussian distribution" )
 * Notice: predict all 100k ratings (no training and testing set)
 * Algorithm: maximization likelihood and EM 
 * Output: matrix factorization U and V, denoising ratings R'= U*V
 * Part II
 * Input:  denoising ratings R'
 * Algorithm: M-distance prediction
 * Output: predictions R''
 */
package algorithm;
import java.io.*;

//import Jama.*;
import datamodel.DataInfor;
//import datamodel.NoiseInfo;
import tool.*;
//import tool.SimpleTool;

public class MLGMDN {
	DataInfor data;
	MBR mbr;

	/**
	 * The mean values of Gaussian distributions.
	 */
	public double[] mu;

	/**
	 * The variances of Gaussian distributions.
	 */
	public double[] sigma;

	/**
	 * The number of users.
	 */
	int numUsers;

	/**
	 * The number of items.
	 */
	int numItems;

	/**
	 * The number of non-zero elements in the training set.
	 */
	int numTrainSize;

	/**
	 * The rank of factorization matrices.
	 */
	public int rank;

	/**
	 * The type of noises.
	 */
	int numNoise;

	/**
	 * The weight of every noise.
	 */
	// public double[] noiseWeight;
	double[] noiseWeight;

	/**
	 * Record the error
	 */
	// public double[] trainingErrors;
	double[] trainingErrors;

	/**
	 * Help computing noiseDistribution. The size is the same as noiseDistribution.
	 */
	// public double[][] logRho;
	double[][] logRho;

	/**
	 * @see logRho
	 */

	double[] logsumexp;
	/**
	 * The log of likelihood
	 */
	double logLikelihood;

	/**
	 * The iteration times of matrix factorization
	 */
	public int decompIterTimes;

	/**
	 * The convergence condition
	 */
	public double tolerance;

	/**
	 * A parameter in EfficientMCL2
	 */
	double[][] Effic_para1;

	/**
	 * The square roots of noiseWeight matrix
	 */
	double[][] sqrtW;

	/**
	 * The two matrices of factorization results
	 */
	public double[][] factor_U;

	/**
	 * The two matrices of factorization results
	 */
	public double[][] factor_V;

	/**
	 * The noise distribution along the training set. The number of columns is the
	 * number of noise.
	 */
	public double[][] noiseDistribution;

	/**
	 * Record the likelihood of each iteration
	 */
	double[] llh;

	/**
	 * The predicted full rating matrix.
	 */
	static double[][] predictions;

	/**
	 * The denoising rating matrix.
	 */
	public static double[][] denoisingMatrix;

	/**
	 **********************
	 * @param paraData
	 ********************** 
	 */
	public MLGMDN(DataInfor paraData) {
		data = paraData;
		initialize();
	}

	/**
	 ***********************
	 * Initialize parameters
	 ***********************
	 */
	public void initialize() {
		factor_U = new double[numUsers][rank];
		factor_V = new double[numItems][rank];

		numUsers = data.getNumUsers();
		numItems = data.getNumItems();

		numTrainSize = data.getTrainingSize();
		rank = data.getLowRank();

		factor_U = data.subU;
		factor_V = data.subV;

		numNoise = data.getNumNoise();
		// Share the same space of data.
		noiseDistribution = data.getNoiseDistribution();
		mu = new double[numNoise];
		sigma = new double[numNoise];
		noiseWeight = new double[numNoise];

		trainingErrors = new double[numTrainSize];
		logLikelihood = 0;
		decompIterTimes = 50;
		tolerance = 1e-8;
		Effic_para1 = new double[numUsers][numItems];
		sqrtW = new double[numUsers][numItems];

		for (int i = 0; i < sigma.length; i++) {
			sigma[i] = Math.random();
			System.out.println("sigma" + i + ":" + sigma[i]);
			mu[i] = 0;
			noiseWeight[i] = data.noiseWeight[i];
			// System.out.printf("sigma %d: %f\r\n", i,sigma[i]);
		} // of for i

		logRho = new double[numTrainSize][numNoise];// 640*3
	}// of initialize

	/**
	 * ********************* * Initialize the parameters of Gaussian distributions
	 * **********************
	 */
	// public void setModel() {
	// for (int i = 0; i < sigma.length; i++) {
	// sigma[i] = Math.random();
	// System.out.println("sigma" + i + ":" + sigma[i]);
	// mu[i] = 0;
	// noiseWeight[i] = data.noiseWeight[i];
	// // System.out.printf("sigma %d: %f\r\n", i,sigma[i]);
	// } // of for i
	// }// of setSigma

	/**
	 ***********************
	 * Compute the error on non-zero points of the training set. Note that the error
	 * can be either positive or negative.
	 * 
	 * @return the train errors in an array with size of the training set.
	 ***********************
	 */
	public double[] computeNonZeroError() {
		double[][] tempX = MatrixOpr.Matrix_Mult(factor_U, factor_V);
		int tempCount = 0;

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < numItems; j++) {
				if (data.nonZeroTrainIndexMatrix[i][j] != 0) {
					trainingErrors[tempCount] = data.trainMatrix[i][j] - tempX[i][j];
					tempCount++;
				} // of if
			} // of for j
		} // of for i

		// System.out.println("The length of trainingErrors: " +
		// trainingErrors.length);
		return trainingErrors;
	}// of computeNonZeroError

	/**
	 ***********************
	 * The E step of the EM algorithm. Compute the expected noise for each data
	 * point Recompute logRho, noiseDistribution and logLikelihood. Update the
	 * noiseDistribution
	 * 
	 * @return noiseDistribution
	 ***********************
	 */
	public double[][] expectation() {
		// double[][] R = new double[numTrainSize][numNoise];
		double[] tempVector = null;// = new double[numTrainSize];

		// logRho = new double[numTrainSize][numNoise];// 640*3

		// Step 1. logRho(:,i) = logGaussPdf(X,mu(i),Sigma(i));
		for (int j = 0; j < logRho[0].length; j++) {
			tempVector = logGaussPdf(trainingErrors, mu[j], sigma[j]);
			for (int i = 0; i < logRho.length; i++) {
				logRho[i][j] = tempVector[i];
			} // for i
		} // for j

		// System.out.println("logRho");
		// SimpleTool.printMatrix(logRho);

		// Step 2. logRho = bsxfun(@plus,logRho,log(w));
		for (int i = 0; i < logRho.length; i++) {
			for (int j = 0; j < logRho[0].length; j++) {
				logRho[i][j] += Math.log(noiseWeight[j]);
			} // of for j
		} // of for i

		// System.out.println("logRho+logw:");
		// SimpleTool.printMatrix(logRho);

		// Step 3. T = logsumexp(logRho,2);
		computeLogSumExp();

		// System.out.println("Print logRho again:");
		// SimpleTool.printMatrix(logRho);

		logLikelihood = 0;
		// Step 4. llh = sum(T)/n; % loglikelihood
		for (int i = 0; i < logsumexp.length; i++) {
			logLikelihood += logsumexp[i];
		} // of for i
		logLikelihood = logLikelihood / numTrainSize;

		// Step 5.logR = bsxfun(@minus,logRho,T);
		// Step 6. R = exp(logR);
		for (int i = 0; i < logRho.length; i++) {
			for (int j = 0; j < logRho[0].length; j++) {
				noiseDistribution[i][j] = Math.exp(logRho[i][j] - logsumexp[i]);
			} // of for j
		} // of for i
		System.out.println("logR:");
		// SimpleTool.printMatrix(logR);

		return noiseDistribution;
	}// of expectation

	/**
	 ******************* 
	 * Iterate Expectation step and Maximization step. The core code of the
	 * algorithm
	 ******************* 
	 */
	// public void iterationEM(int paramaxIterTimes) {
	public void iterationEM(int paraMaxIterTimes) {
		// int tempmaxIterTimes = raparamaxIterTimes
		boolean converged = false;
		int tempIterations = 0;
		llh = new double[paraMaxIterTimes];
		double templlh1;
		double templlh2;

		// subU and subV are already initialized
		computeNonZeroError();

		// SimpleTool.printDoubleArray(tempEm.trainingErrors);
		expectation();

		// Initialization

		llh[0] = logLikelihood;// before iterations
		// If converged, do not repeat more
		while (converged != true && tempIterations < paraMaxIterTimes) {
			tempIterations++;
			// ******* M Step 1************
			updateNoiseParameters();
			// ******* M Step 1************

			// ******* E Step 1 compute the noise distribution************
			expectation();
			// templlh1 = logLikelihood;
			// llh[tempIterations] = templlh1;
			llh[tempIterations] = logLikelihood;
			// ******* E Step 1************

			// ******* M Step 2************
			// Compute subU and subV according to the noise distribution
			maximizationW();

			// Prepare for the noise part
			computeNonZeroError();
			// ******* M Step 2************

			// ******* E Step 2************
			expectation();
			// templlh2 = logLikelihood;
			llh[tempIterations] = logLikelihood;

			// ******* E Step 2************
			if (llh[tempIterations] - llh[tempIterations - 1] < tolerance * Math.abs(llh[tempIterations])) {
				converged = true;
			} // Of if
		} // of while
		System.out.println("Acutual iteration time: " + tempIterations);
	}// of iteration

	/**
	 ***********************
	 * Compute logsumexp using logRho.<br>
	 * Suppose that k = 3 and \log a_2 is the maximal value among {\log a_1, \log
	 * a_2, \log a_3} $\log (a_1 + a_2 + a3) = \log(e^{\log a_1 - \log a_2} + e^0$ $
	 * + e^{\log a_3 - \log a_2}) + \log a_2$ The purpose is to avoid computing
	 * $a_1$, $a_2$, and $a_3$ which may overflow (too close to 0).
	 * 
	 * @return logsumexp Each element of the vector is obtained using one row of
	 *         logRho
	 ***********************
	 */
	public double[] computeLogSumExp() {
		/*
		 * Step 1. y = max(x,[],dim) Compute the max value of every row in Matrix
		 * logRho.
		 */
		double[][] tempXX = new double[logRho.length][logRho[0].length];
		double[] tempRowMax = new double[logRho.length];
		for (int i = 0; i < logRho.length; i++) {
			tempRowMax[i] = logRho[i][0];
			for (int j = 1; j < logRho[0].length; j++) {
				if (tempRowMax[i] < logRho[i][j]) {
					tempRowMax[i] = logRho[i][j];
				} // of if
			} // of for j
		} // of for i

		// Step 2. Compute x = bsxfun(@minus,x,y)
		for (int j = 0; j < logRho[0].length; j++) {
			for (int i = 0; i < logRho.length; i++) {
				tempXX[i][j] = logRho[i][j] - tempRowMax[i];
			} // of for j
		} // of for i

		// Step 3. s = y +log(sum(exp(x),dim)),dim=2,compute the sum of every
		// row.
		double[] tempSum = new double[numTrainSize];
		for (int i = 0; i < logRho.length; i++) {
			tempSum[i] = 0;
			for (int j = 0; j < logRho[0].length; j++) {
				tempSum[i] += Math.exp(tempXX[i][j]);
			} // of for j
			tempSum[i] = Math.log(tempSum[i]);
		} // of for i
		logsumexp = tool.MatrixOpr.Vector_Add(tempRowMax, tempSum);
		System.out.println("logsumexp:");
		// SimpleTool.printDoubleArray(logsumexp);

		return logsumexp;
	}// of computeLogSumExp

	/**
	 ***********************
	 * Compute logsumexp using logRho.<br>
	 * The expression in Latex
	 * 
	 * @return logsumexp Each element of the vector is obtained using one row of
	 *         logRho
	 ***********************
	 */
	public double[] computeLogSumExpV2() {

		return logsumexp;
	}// Of computeLogSumExpV2

	/**
	 * logGaussPdf is to compute ln(gauss probability funtion), and divided into
	 * three parts. Part1: from Step1 to Step4 Part2 and Part3: Step5 Compute the
	 * log of probability density function x -> ln f(x)
	 * 
	 * @param paraError
	 *            each element represents the error
	 * 
	 * @return the probability density of each error.
	 */
	public double[] logGaussPdf(double[] paraErrors, double paraMu, double paraSigma) {

		double sigma_Decomp;
		// Make a copy to avoid changing the vector
		double[] tempErrors = new double[paraErrors.length];
		for (int i = 0; i < tempErrors.length; i++) {
			tempErrors[i] = paraErrors[i];
		} // Of for i

		double[] templogGaussPdf = new double[tempErrors.length];
		// Step 1. paraError subtracts paraMu. X = bsxfun(@minus,X,mu);
		for (int i = 0; i < paraErrors.length; i++) {
			tempErrors[i] = paraErrors[i] - paraMu;
		} // of for i

		/*
		 * Step 2: CholeskyDecomposition of paraSigma. For convenience, employ "sqrt"
		 * instead of it. [U,p] = chol(Sigma);
		 */
		sigma_Decomp = Math.sqrt(paraSigma);

		// Step 3. Compute Q = U' \ X.

		double[] tempQ = new double[tempErrors.length];
		for (int i = 0; i < tempErrors.length; i++) {
			tempQ[i] = tempErrors[i] / sigma_Decomp;
		} // of for i
			// SimpleTool.printDoubleArray(tempQ);

		// Step 4. Compute dot(Q, Q, 1)
		double[] tempPart1;

		tempPart1 = MatrixOpr.Vector_DotMult(tempQ, tempQ);

		// System.out.println("fajfdlafj" + tempQ.length);
		// SimpleTool.printDoubleArray(tempPart1);

		// Step 5. Compute c = d*log(2*pi)+2*sum(log(diag(U)));
		// log(diag(U)):U is a double number, equal to sigma_Decomp
		double tempPart2 = Math.log(2 * Math.PI);
		double tempPart3 = 2 * Math.log(sigma_Decomp);

		// Step 6. Compute logGaussPdf
		for (int i = 0; i < tempErrors.length; i++) {
			templogGaussPdf[i] = -(tempPart1[i] + tempPart2 + tempPart3) / 2;
		} // of for i
		System.out.println("templogGaussPdf:");
		// SimpleTool.printDoubleArray(templogGaussPdf);
		return templogGaussPdf;
	}// of logGaussPdf

	/**
	 ***********************
	 * Update the parameters of Gaussian distribution for the noise. Also called
	 * "maximizationModel"
	 ***********************
	 */
	public void updateNoiseParameters() {
		// update noise parameters
		// double[][] R = new double[numTrainSize][numNoise];
		double[] tempSum;

		double[] paraWeight = new double[noiseWeight.length];
		double[] tempXX = new double[numTrainSize];
		// double[][] tempXX_transp = new double[R[0].length][R.length];
		double[] tempXXX = new double[numNoise];
		double[] tempsqrtR = new double[numTrainSize];
		// R = noiseDistribution;
		/*
		 * Step 1. Recompute noiseWeight nk = sum(R,1); w = nk/size(R,1); Sigma =
		 * zeros(1,k); sqrtR = sqrt(R);
		 */
		tempSum = tool.MatrixOpr.SumbyCol(noiseDistribution);

		/*
		 * Step 2. Recompute mu and sigma mu = zeros(1,k);%fix mu to zero
		 */
		for (int i = 0; i < noiseWeight.length; i++) {
			mu[i] = 0;
			sigma[i] = 0;
			paraWeight[i] = tempSum[i] / numTrainSize;
		} // of for i

		// Step 3. Update sigma
		for (int j = 0; j < numNoise; j++) {
			for (int i = 0; i < numTrainSize; i++) {
				tempsqrtR[i] = Math.sqrt(noiseDistribution[i][j]);
				tempXX[i] = (trainingErrors[i] - mu[j]) * tempsqrtR[i];
			} // of for i
			tempXXX[j] = tool.MatrixOpr.Vector_Mult(tempXX, tempXX);
			sigma[j] = tempXXX[j] / tempSum[j] + 1e-6;
		} // of for j

		noiseWeight = paraWeight;
		System.out.println("sigma is updated:");
		SimpleTool.printDoubleArray(sigma);
	}// of updateNoiseParameters

	/**
	 ***********************
	 * Compute two subMatrices of factorization. Prepare/initialize for
	 * EfficientMCL2()
	 ***********************
	 */
	public void maximizationW() {
		double[][] tempW = new double[numUsers][numItems];
		double[][] tempC = new double[numUsers][numItems];
		// double[][] R;
		// double[] tempMu;
		// double[] tempSigma;

		// R = noiseDistribution;
		// tempMu = mu;
		// tempSigma = sigma;

		/*
		 * Step 1. for j = 1:k W(IND) = W(IND) + R(:,j)/(2*Sigma(j)); C(IND) = C(IND) +
		 * R(:,j)*mu(j)/(2*Sigma(j)); end
		 */
		for (int k = 0; k < noiseDistribution[0].length; k++) {
			int tempCnt = 0;
			for (int i = 0; i < numUsers; i++) {
				for (int j = 0; j < numItems; j++) {
					if (data.nonZeroTrainIndexMatrix[i][j] != 0 && sigma[k] != 0) {
						tempW[i][j] += noiseDistribution[tempCnt][k] / (2 * sigma[k]);
						tempC[i][j] += noiseDistribution[tempCnt][k] * mu[k] / (2 * sigma[k]);// always equal to zero
						tempCnt++;
					} // of if
				} // of for i
			} // of for j
		} // of for k

		tempC = tool.MatrixOpr.matrixDotDivision(tempC, tempW);// tempC = 0

		// Step 2. Compute Inx - C and sqrt(W)
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < numItems; j++) {
				Effic_para1[i][j] = data.trainMatrix[i][j] - tempC[i][j];
				sqrtW[i][j] = Math.sqrt(tempW[i][j]);// sqrtW
			} // of for j
		} // of for i

		EfficientMCL2();
	}// of MaximazationW

	/**
	 ***********************
	 * A main algorithm. The low rank matrices are updated decompIterTimes times.
	 ***********************
	 */
	public void EfficientMCL2() {
		int[] randperm;// order randomly

		double[][] tempMulti;
		double[][] tempRegul = new double[numUsers][numItems];// Regulation
		double[][] tempRegul_transp;
		double[][] tempsqrtW_transp;

		double[][] tempPart;

		double[] paraColVec;
		double[] paraRowVec;

		double[][] tempNorm;
		double[] tempVector1 = new double[rank];
		double[] tempVector2 = new double[rank];
		double[] tempVector3 = new double[rank];
		double[][] tempDiag_Sqrt = new double[tempVector1.length][tempVector2.length];
		double[][] tempDiag_U = new double[tempVector1.length][tempVector1.length];
		double[][] tempDiag_V = new double[tempVector2.length][tempVector2.length];
		double[][] tempOutU;
		double[][] tempOutV;
		double[][] tempOutU_transp;
		double[][] tempOutV_transp;

		double[][] paraSubU;
		double[][] paraSubV;
		double[][] paraSubU_transp = new double[numUsers][rank];
		double[][] paraSubV_transp = new double[numItems][rank];
		// paraSubU: InU in EfficientMCL2, single direction, not return
		// System.out.println("**************data.subU************");
		// printMatrix(data.subU);
		// System.out.println("**************data.subV************");
		// printMatrix(data.subV);

		paraSubU = data.subU;
		paraSubV = data.subV;

		tempOutU = data.subU;
		tempOutV = data.subV;

		for (int i = 0; i < decompIterTimes; i++) {
			randperm = SimpleTool.generateRandomSequence(rank);
			for (int j = 0; j < randperm.length; j++) {// 1-4 disorder
				// Step 1. TX = Matrix - OutU*OutV' + OutU(:,j)*OutV(:,j)'

				tempMulti = tool.MatrixOpr.Matrix_Mult(tempOutU, tempOutV);// =
																			// OutU*OutV'

				tempOutU_transp = tool.MatrixOpr.Matrix_Transpose(tempOutU);// 4*40

				tempOutV_transp = tool.MatrixOpr.Matrix_Transpose(tempOutV);// 4*20

				int tempColInd = randperm[j];
				paraColVec = tempOutU_transp[tempColInd];
				paraRowVec = tempOutV_transp[tempColInd];

				tempPart = tool.MatrixOpr.ColVec_Multi_RowVec(paraColVec, paraRowVec);// 943*1682
				for (int tempi = 0; tempi < tempRegul.length; tempi++) {
					for (int tempj = 0; tempj < tempRegul[0].length; tempj++) {
						tempRegul[tempi][tempj] = Effic_para1[tempi][tempj] - tempMulti[tempi][tempj]
								+ tempPart[tempi][tempj];
					} // of for tempj
				} // of for tempi

				// Transpose tempU and tempV for the following computation.

				tempRegul_transp = tool.MatrixOpr.Matrix_Transpose(tempRegul);
				tempsqrtW_transp = tool.MatrixOpr.Matrix_Transpose(sqrtW);// 20*40

				/*
				 * Step 2. u = InU(:,j); OutV(:,j) = optimMCL2(TX,W,u) OutU(:,j) =
				 * optimMCL2(TX',W',OutV(:,j))
				 */
				// System.out.printf("Iteration time: %d,tempColInd:%d\r\n",
				// i,tempColInd);

				paraSubU_transp = tool.MatrixOpr.Matrix_Transpose(paraSubU);// 4*40

				paraSubV_transp = tool.MatrixOpr.Matrix_Transpose(paraSubV);// 4*20

				tempOutV_transp[tempColInd] = optimMCL2(tempRegul, sqrtW, paraSubU_transp[tempColInd]);
				// System.out.println("tempColInd:"+tempColInd);
				// System.out.println("**************tempOutV_transp************");
				// printMatrix(tempOutV_transp);

				tempOutU_transp[tempColInd] = optimMCL2(tempRegul_transp, tempsqrtW_transp,
						tempOutV_transp[tempColInd]);
				// System.out.println("**************tempOutU_transp************");
				// printMatrix(tempOutU_transp);

				tempOutU = tool.MatrixOpr.Matrix_Transpose(tempOutU_transp);
				tempOutV = tool.MatrixOpr.Matrix_Transpose(tempOutV_transp);
			} // of for j

			paraSubU = tool.MatrixOpr.Matrix_Transpose(paraSubU_transp);
			paraSubV = tool.MatrixOpr.Matrix_Transpose(paraSubV_transp);

			// Step 3. Compute norm

			tempNorm = tool.MatrixOpr.Matrix_Sub(paraSubU, tempOutU);
			tempNorm = tool.MatrixOpr.Matrix_DotMult(tempNorm, tempNorm);
			double tempSum = tool.MatrixOpr.Matrix_Sum(tempNorm);
			// System.out.printf("Norm is %f\r\n",Math.sqrt(tempSum) );
			if (Math.sqrt(tempSum) < tolerance) {
				break;
			} else {
				paraSubU = tempOutU;
			} // of if
		} // of for i

		/*
		 * Step 4. Nu = sqrt(sum(OutU.^2))'; Nv = sqrt(sum(OutV.^2))'; No =diag(Nu.*Nv);
		 * OutU = OutU*diag(1./Nu)*sqrt(No); OutV = OutV*diag(1./Nv)*sqrt(No);
		 */
		tempVector1 = tool.MatrixOpr.SumbyCol(tool.MatrixOpr.Matrix_DotMult(tempOutU, tempOutU));// 1*4
		tempVector2 = tool.MatrixOpr.SumbyCol(tool.MatrixOpr.Matrix_DotMult(tempOutV, tempOutV));// 1*4
		for (int i = 0; i < tempVector1.length; i++) {
			tempVector1[i] = Math.sqrt(tempVector1[i]);
			tempVector2[i] = Math.sqrt(tempVector2[i]);
		} // of for i
		tempVector3 = tool.MatrixOpr.Vector_DotMult(tempVector1, tempVector2);

		for (int i = 0; i < tempDiag_Sqrt.length; i++) {
			tempDiag_Sqrt[i][i] = Math.sqrt(tempVector3[i]);// sqrt(No)
			tempDiag_U[i][i] = 1.0 / tempVector1[i];// diag(1./Nu)
			tempDiag_V[i][i] = 1.0 / tempVector2[i];// diag(1./Nv)
		} // of for i

		tempOutU = tool.MatrixOpr.Matrix_Mult(tempOutU, tool.MatrixOpr.Matrix_Mult(tempDiag_U, tempDiag_Sqrt));
		tempOutV = tool.MatrixOpr.Matrix_Mult(tempOutV, tool.MatrixOpr.Matrix_Mult(tempDiag_V, tempDiag_Sqrt));
		factor_U = tempOutU;
		factor_V = tempOutV;
		// System.out.println("*************************Factorization
		// U*************************");
		// SimpleTool.printMatrix(factor_U);
		// System.out.println("*************************Factorization
		// V*************************");
		// SimpleTool.printMatrix(factor_V);
	}// of EfficientMCL2;

	/**
	 ***********************
	 * Compute a column vector.
	 * 
	 * @param paraMatrix
	 * @return The column vector indicating
	 * @see #EfficientMCL2()
	 ***********************
	 */
	public double[] optimMCL2(double[][] paraMatrix, double[][] parasqrtW, double[] paraVector) {
		double[][] tempX;
		double[][] tempXX;
		double[] resultOptimVec;
		double[][] tempVec2Matrix;
		double[] tempUp;
		double[] tempDown;

		// Step 1. TX = W.*Matrix, size: equal to W or Matrix,
		tempX = tool.MatrixOpr.Matrix_DotMult(parasqrtW, paraMatrix);

		// Step 3. U = u*ones(1,n), size: paraVector.length
		tempVec2Matrix = tool.MatrixOpr.Vector2Matrix(paraVector, paraMatrix[0].length);

		// Step 4. U = W.* U; size:equal to W,
		tempXX = tool.MatrixOpr.Matrix_DotMult(parasqrtW, tempVec2Matrix);

		// Step 5. up = sum(TX.*U), size: the column number of TX
		tempUp = tool.MatrixOpr.SumbyCol(tool.MatrixOpr.Matrix_DotMult(tempX, tempXX));

		// Step 6. down = sum (U.* U), size: the column number of U (or TX)
		tempDown = tool.MatrixOpr.SumbyCol(tool.MatrixOpr.Matrix_DotMult(tempXX, tempXX));

		// Step 7. v = up ./ down
		resultOptimVec = tool.MatrixOpr.Vector_DotDiv(tempUp, tempDown);
		return resultOptimVec;
	}// of optimMCL2

	/**
	 ***********************
	 * Print the matrix
	 ***********************
	 */
	public void printMatrix(double[][] paraMatrix) {
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				System.out.printf("|%8.5f|", paraMatrix[i][j]);
			} // Of for j
			System.out.print("*");
			System.out.println();
		} // Of for i
	}// of printMatrix

	/**
	 ***********************
	 * Compute which distribution the noise on every rating belongs to.
	 ***********************
	 */
	public void printNoiseDistribution() {
		int[] belongVector = new int[noiseDistribution.length];
		double[] tempRowMax = new double[noiseDistribution.length];
		for (int i = 0; i < noiseDistribution.length; i++) {
			tempRowMax[i] = noiseDistribution[i][0];
			belongVector[i] = 0;
			for (int j = 1; j < noiseDistribution[0].length; j++) {
				if (tempRowMax[i] < noiseDistribution[i][j]) {
					belongVector[i] = j;
				} // of if
			} // of for j
		} // of for i

		System.out.println("The max probability of each row in R");
		for (int i = 0; i < 5; i++) {
			System.out.printf("%5d", belongVector[i]);
		} // of for i
		System.out.println("");
		System.out.println("The indices of data set");
		for (int i = 0; i < 5; i++) {
			int tempCol = data.getTrainingDataColumn(i);
			int tempRow = data.getTrainingDataRow(i);
			System.out.printf("(Row:%d, Col:%d)\t", tempRow, tempCol);
		} // of for i
		System.out.println("");
		System.out.println("Sigma");
		SimpleTool.printDoubleArray(sigma);
	}// of printNoiseDistribution

	/**
	 ********************** 
	 * Compute the predicted rating matrix.
	 ********************** 
	 */
	public void computePredictions() {
		predictions = tool.MatrixOpr.Matrix_Mult(factor_U, factor_V);
		System.out.println("The size of paraMatrix_Multi:" + predictions.length + "," + predictions[0].length);
		
		predictions = tool.MatrixOpr.Matrix_DotMult(predictions, data.nonZeroIndexMatrix);
		//predictions = tool.MatrixOpr.Add_MatrixandNumber(predictions, data.meanRatingOfTrain);
		for (int i = 0; i < predictions.length; i++) {
			for (int j = 0; j < predictions[0].length; j++) {
				if(data.nonZeroIndexMatrix[i][j] == 1) {
				predictions[i][j] += data.meanRatingOfTrain;
				if (predictions[i][j] < 1) {
					//if (predictions[i][j] < 1) {
						// System.out.println("Value exceeds bound: "
						// + predictions[i][j]);
						predictions[i][j] = 1;
					} // of if
					if (predictions[i][j] > 5) {
						predictions[i][j] = 5;
					} // of if
				}//Of if
			} // of for j
		} // of for i
			
		// SimpleTool.printMatrix(paraMatrix_Multi);
//		for (int i = 0; i < predictions.length; i++) {
//			for (int j = 0; j < predictions[0].length; j++) {
//				// if (data.testMatrix[i][j] < 1e-6)
//				// continue;
//				if (predictions[i][j] > 0 && predictions[i][j] < 1) {
//				//if (predictions[i][j] < 1) {
//					// System.out.println("Value exceeds bound: "
//					// + predictions[i][j]);
//					predictions[i][j] = 1;
//				} // of if
//				if (predictions[i][j] > 5) {
//					predictions[i][j] = 5;
//				} // of if
//			} // of for j
//		} // of for i
//		 SimpleTool.printMatrix(predictions);
	}// Of computePredictions

	/**
	 ********************** 
	 * Compute the distance between the predicted matrix and the original one.
	 ********************** 
	 */
	public void printErrors() {
		double tempDistanceofTrain = 0;
		double tempDistanceofTest = 0;
//		double tempSubspace_U = 0;
//		double tempSubspace_V = 0;
		double trainMAE = 0;
		double trainRMSE = 0;
		double testMAE = 0;
		double testRMSE = 0;
		double totalMAE = 0;
		double totalRMSE = 0;
		double totalVariance = 0;
		// double[][] paraMatrix_Multi = new double[numUsers][numItems];
		double[][] tempMatrix_Multi_TrainNonzero = new double[numUsers][numItems];
		double[][] tempMatrix_Multi_TestNonzero = new double[numUsers][numItems];
		double[][] tempTest_SubValue = new double[numUsers][numItems];
		double[][] tempTrain_SubValue = new double[numUsers][numItems];
		double[][] tempTrainSquare = new double[numUsers][numItems];
		double[][] tempTestSquare = new double[numUsers][numItems];
		double[][] tempTrainAbs = new double[numUsers][numItems];
		double[][] tempTestAbs = new double[numUsers][numItems];
		//denoisingMatrix = new double[numUsers][numItems];
		// for (int i = 0; i < predictions.length; i++) {
		// for (int j = 0; j < predictions[0].length; j++) {
		// if (data.trainMatrix[i][j] > 1e-6) {
		// tempMatrix_Multi_TrainNonzero[i][j] = predictions[i][j];
		// }
		// } // of for j
		// } // of for i
		tempMatrix_Multi_TrainNonzero = tool.MatrixOpr.Matrix_DotMult(predictions, data.nonZeroTrainIndexMatrix);

		tempMatrix_Multi_TestNonzero = tool.MatrixOpr.Matrix_DotMult(predictions, data.nonZeroTestIndexMatrix);

//		denoisingMatrix = tool.MatrixOpr.Matrix_Add(tempMatrix_Multi_TrainNonzero, tempMatrix_Multi_TestNonzero);
//		System.out.println("denoisingMatrix:");

		// SimpleTool.printMatrix(denoisingMatrix);
		// SimpleTool.printMatrix(tempMatrix_Multi_TestNonzero);
		tempTest_SubValue = tool.MatrixOpr.Matrix_Sub(tempMatrix_Multi_TestNonzero, data.testMatrix);
		tempTrain_SubValue = tool.MatrixOpr.Matrix_Sub(tempMatrix_Multi_TrainNonzero, data.trainMatrix);
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < numItems; j++) {

				trainMAE += Math.abs(tempMatrix_Multi_TrainNonzero[i][j] - data.trainMatrix[i][j]);
				testMAE += Math.abs(tempMatrix_Multi_TestNonzero[i][j] - data.testMatrix[i][j]);

				tempDistanceofTrain += (tempMatrix_Multi_TrainNonzero[i][j] - data.trainMatrix[i][j])
						* (tempMatrix_Multi_TrainNonzero[i][j] - data.trainMatrix[i][j]);
				tempDistanceofTest += (tempMatrix_Multi_TestNonzero[i][j] - data.testMatrix[i][j])
						* (tempMatrix_Multi_TestNonzero[i][j] - data.testMatrix[i][j]);

			} // of for j
		} // of for i
		totalMAE = trainMAE + testMAE;
		trainMAE /= numTrainSize;
		trainRMSE = Math.sqrt(tempDistanceofTrain / numTrainSize);
		testMAE /= data.getTestingSize();
		testRMSE = Math.sqrt(tempDistanceofTest / data.getTestingSize());
		totalMAE /= (numTrainSize + data.getTestingSize());
		totalRMSE = Math.sqrt((tempDistanceofTrain + tempDistanceofTest) / (numTrainSize + data.getTestingSize()));

		System.out.printf("Train and Test MAE:%8.2f, %8.2f\r\n", trainMAE, testMAE);
		System.out.printf("Train and Test RMSE:%8.2f, %8.2f\r\n", trainRMSE, testRMSE);
		System.out.printf("The total MAE and RMSE:%8.2f, %8.2f\r\n", totalMAE, totalRMSE);
		// tempTrainSquare = tool.MatrixOpr.Matrix_DotMult(tempTrain_SubValue,
		// tempTrain_SubValue);
		// tempTestSquare = tool.MatrixOpr.Matrix_DotMult(tempTest_SubValue,
		// tempTest_SubValue);
		tempDistanceofTrain = Math.sqrt(tempDistanceofTrain);
		tempDistanceofTest = Math.sqrt(tempDistanceofTest);
		System.out.printf("The euclidean metric between two matrices (Train and Test):%f, %f\r\n", tempDistanceofTrain,
				tempDistanceofTest);

		// System.out.println("Subspace U");
		// printMatrix( tool.MatrixOpr.Matrix_Subspace(factor_U, data.subU));
		// System.out.println("Subspace V");
		// printMatrix( tool.MatrixOpr.Matrix_Subspace(factor_V, data.subV));
		// System.out.printf("The distance between two matrices:%f\r\n",tempDistance);
		// System.out.printf("Subspace U:%f , Subspace
		// V:%f\r\n",tempSubspace_U,tempSubspace_V);

	}// Of printErrors

	public void observeDenoisingMatrix(double paraMatrix[][]) {
		double tempSum, tempMAE = 0, tempVariance = 0;
		double tempAve;
		double tempTotalNum = numTrainSize + data.getTestingSize();
		tempSum = tool.MatrixOpr.Matrix_Sum(paraMatrix);
		tempAve = tempSum / tempTotalNum;
		System.out.printf("The mean value of R' (denoisingMatrix) is %8.2f\r\n", tempAve);
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[0].length; j++) {
				if (paraMatrix[i][j] > 1e-6) {
					tempMAE += Math.abs(paraMatrix[i][j] - tempAve);
					tempVariance += (paraMatrix[i][j] - tempAve) * (paraMatrix[i][j] - tempAve);
				} // Of if
			} // Of for j
		} // Of for i
		tempMAE /= tempTotalNum;
		tempVariance /= tempTotalNum;
		System.out.printf("The variance  of R' (denoisingMatrix) is %8.2f\r\n", tempVariance);
		System.out.printf("The MAE of R' (denoisingMatrix) between its mean value: %8.2f\r\n", tempMAE);
	
	}// Of observePredictions

	
	public void observeRR(double[] paraPred) {

		double tempMAE = 0;
		double tempAve = 0;
		double tempVariance = 0;
		for (int i = 0; i < paraPred.length; i++) {
			tempAve += paraPred[i];
		} // Of for i
		tempAve /= paraPred.length;
		System.out.printf("The mean value of R'' (after M-distance) is %8.2f\r\n", tempAve);
		for (int i = 0; i < paraPred.length; i++) {
			tempMAE += Math.abs(paraPred[i] - tempAve);
			tempVariance += (paraPred[i] - tempAve) * (paraPred[i] - tempAve);
		} // Of for i
		tempMAE /= paraPred.length;
		tempVariance /= paraPred.length;
		System.out.printf("The variance of R'' (after M-distance) between its mean value: %8.2f\r\n", tempVariance);
		System.out.printf("The MAE of R'' (after M-distance) between its mean value: %8.2f\r\n", tempMAE);
	}// Of observeRR

	public void writeMatrix2Txt1(double[][] paraMatrix) throws IOException {
		
		int[] tempCnt = new int[8];
		for(int i = 0; i < paraMatrix.length; i++) {
			for(int j = 0; j < paraMatrix[0].length; j++) {
				double tempValue = paraMatrix[i][j];
				if(tempValue > 1e-6 && tempValue <=1) {
					tempCnt[0]++;
				}else if(tempValue > 1 && tempValue <= 2) {
					tempCnt[1]++;
				}else if(tempValue > 2 && tempValue <= 2.5) {
					tempCnt[2]++;
				}else if(tempValue > 2.5 && tempValue <= 3) {
					tempCnt[3]++;
				}else if(tempValue > 3 && tempValue <= 3.5) {
					tempCnt[4]++;
				}else if(tempValue > 3.5 && tempValue <= 4) {
					tempCnt[5]++;
				}else if(tempValue > 4 && tempValue <= 4.5) {
					tempCnt[6]++;
				}else if(tempValue > 4.5 && tempValue <= 5) {
					tempCnt[7]++;				
				}
			}//Of for j
		}//Of for i 
		File file11 = new File("ratingsOfFactorizationResult.txt");
		FileWriter out11 = new FileWriter(file11);
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[0].length; j++) {
				if(paraMatrix[i][j] > 1e-6) {
				//out11.write(+paraMatrix[i][j] + ",");
					out11.write(+ paraMatrix[i][j]+"\r\n");
					//out11.write("\r\n");
				}//Of if
//				out11.write("\r\n");
			} // Of for j
		} // Of for i
		out11.close();
		
		File file2 = new File("distributionOfGMMPrediction.txt");
		FileWriter out2 = new FileWriter(file2);
		for (int i = 0; i < tempCnt.length; i++) {
					out2.write(+ tempCnt[i]+"\r\n");
		} // Of for i
		out2.close();
	}// Of writeMatrix2Txt1

	
public void writeMatrix2Txt2(double[] paraArray) throws IOException {
		
		int[] tempCnt = new int[8];
		for(int i = 0; i < paraArray.length; i++) {
//			for(int j =0; j < paraMatrix[0].length; j++) {
				double tempValue = paraArray[i];
//				if(tempValue > 1e-6 && tempValue <=1) {
				if(tempValue <=1) {
					tempCnt[0]++;
				}else if(tempValue > 1 && tempValue <= 2) {
					tempCnt[1]++;
				}else if(tempValue > 2 && tempValue <= 2.5) {
					tempCnt[2]++;
				}else if(tempValue > 2.5 && tempValue <= 3) {
					tempCnt[3]++;
				}else if(tempValue > 3 && tempValue <= 3.5) {
					tempCnt[4]++;
				}else if(tempValue > 3.5 && tempValue <= 4) {
					tempCnt[5]++;
				}else if(tempValue > 4 && tempValue <= 4.5) {
					tempCnt[6]++;
				}else if(tempValue > 4.5 && tempValue <= 5) {
					tempCnt[7]++;				
				}//Of if
//			}//Of for j
		}//Of for i 
		File file11 = new File("ratingsOfMBR.txt");
		FileWriter out11 = new FileWriter(file11);
		for ( int i = 0; i < paraArray.length; i++) {
//			for (int j = 0; j < paraMatrix[0].length; j++) {
//				if(paraArray[i] > 1e-6) {
				//out11.write(+paraMatrix[i][j] + ",");
					out11.write(+ paraArray[i]+"\r\n");
					//out11.write("\r\n");
//				}//Of if
//				out11.write("\r\n");
//			} // Of for j
		} // Of for i
		out11.close();
		
		File file2 = new File("distributionOfMBRPrediction.txt");
		FileWriter out2 = new FileWriter(file2);
		for (int i = 0; i < tempCnt.length; i++) {
					out2.write(+ tempCnt[i]+"\r\n");
		} // Of for i
		out2.close();
	}// Of writeMatrix2Txt2

	/**
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// Prepare data and preprocessing
		String tempFilename = new String("data/u.data");
		DataInfor tempData = new DataInfor(tempFilename, 943, 1682);
		tempData.splitTrainAndTest(0.0, 5);
		tempData.computeTrainingSetAverageRating();
		tempData.recomputeTrainset();// subtract the average of training set
		tempData.computeTrainVector();
		tempData.generateRandomSubMatrix(2);
		tempData.setRandomNoiseDistribution(3);
		tempData.computeWeight();// For the first time. Round 0

		MLGMDN tempEm = new MLGMDN(tempData);
		// tempEm.setModel();

		tempEm.iterationEM(100);
		tempEm.printNoiseDistribution();
		tempData.recoverTrainMatrix();
		tempEm.computePredictions();
		tempEm.printErrors();
		System.out.println("llh");
		SimpleTool.printDoubleArray(tempEm.llh);
		tempEm.writeMatrix2Txt1(predictions);
		tempEm.observeDenoisingMatrix(predictions);
		
		
		
		MBR tempMdist = new MBR(predictions, 943, 1682, 100000);
		tempMdist.setRadius(0.3);
		tempMdist.leaveOneOutPrediction();
		System.out.printf("The MAE and RMSE of M distance between R' and R'': %8.4f,%8.4f\r\n", tempMdist.computeMAE(),
				tempMdist.computeRSME());
		//tempEm.observeDenoisingMatrix(predictions);
		tempEm.observeRR(tempMdist.getPrediction());
		tempEm.writeMatrix2Txt2(tempMdist.getPrediction());
	}// of main
}// of class EM
