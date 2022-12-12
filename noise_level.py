"""
Liu X, Tanaka M, Okutomi M. Single-image noise level estimation for blind denoising[J]. IEEE transactions on image processing, 2013, 22(12): 5226-5237.
"""

import cv2
import numpy as np
import sklearn.feature_extraction.image as skf
import scipy.stats as sp
import math
import os
import xlsxwriter as xw


def generateToeplitz(derivativeMatrix, PS):
    numRow, numCol = np.shape(derivativeMatrix)
    Temp = np.zeros([(PS - numRow + 1) * (PS - numCol + 1) + 1, (PS * PS) + 1])
    T = np.zeros([(PS - numRow + 1) * (PS - numCol + 1), (PS * PS)])
    rowNum = 1
    for i in range(1, PS - numRow + 2):
        for j in range(1, PS - numCol + 2):
            for p in range(1, numRow + 1):
                Temp[rowNum][((i - 1 + p - 1) * PS + (j - 1) + 1):((i - 1 + p - 1) * PS + (j - 1) + 2 + numCol - 1)] = \
                    derivativeMatrix[p - 1][:]
            rowNum += 1
    T = Temp[1:, 1:]
    return T


def computeVariance(patchCollection, patchSize):
    covOfImage = np.zeros([patchSize * patchSize, patchSize * patchSize])
    _, numP = np.shape(patchCollection)
    covOfImage = patchCollection.dot(patchCollection.T) / (numP - 1)
    Sigma = np.around(covOfImage, decimals=2)
    eigenValue, vect = np.linalg.eig(Sigma)
    varValue = np.min(eigenValue)
    varValue = np.around(varValue, decimals=2)
    return varValue


def noiseLevelEstimation(noisyImage, patchSize=7, confidenceLevel=1 - 1e-6, numIteration=10):
    """
    Input:
        pathI : Path for noisy RGB image
        patchSize= Image patch size
        confidenceLevel:  select close to 1
        numIteration : Max number of iterations
    """
    noisyImage = np.around(noisyImage, decimals=2)
    if (patchSize < 3):
        print("Patch size must be greater than or equal to 3")
        return None
    # Horizantal and vertical derivative operators
    horizontalKernel = np.ones((1, 3), np.float32)
    horizontalKernel[0][0], horizontalKernel[0][1], horizontalKernel[0][2] = -1 / 2, 0, 1 / 2
    verticalKernel = np.ones((3, 1), np.float32)
    verticalKernel[0][0], verticalKernel[1][0], verticalKernel[2][0] = -1 / 2, 0, 1 / 2
    # Toeplitz form of derivative operators
    Dh = generateToeplitz(horizontalKernel, patchSize)
    Dv = generateToeplitz(verticalKernel, patchSize)
    DD = ((Dh.T).dot(Dh)) + ((Dv.T).dot(Dv))
    # Inverse gamma CDF computation for given confidence interval
    k1 = np.matrix.trace(DD)
    r = np.linalg.matrix_rank(DD)
    inverseGammaCDF = 2 * sp.gamma.ppf(confidenceLevel, a=((float(r)) / 2), loc=0, scale=k1 / float(r))
    inverseGammaCDF = np.around(inverseGammaCDF, decimals=2)
    thresold = np.zeros([numIteration, 1])
    var = np.zeros([numIteration, 1])
    # Low rank patch selection(Iterative Framework)
    estimatedVariance = 0
    for i in range(numIteration):
        if (i == 0):
            # variance computation and patch collection for first iteration
            patchesofImage = skf.extract_patches_2d(noisyImage, (patchSize, patchSize))
            MAX_noPatches, _, _ = np.shape(patchesofImage)
            Ipatches = np.zeros([patchSize * patchSize, MAX_noPatches], dtype=float)
            for n in range(MAX_noPatches):
                a = patchesofImage[n, :, :]
                aa = a.reshape(patchSize * patchSize, 1)
                Ipatches[:, n] = aa[:, 0]
            patchCollection = Ipatches

        var[i, 0] = computeVariance(patchCollection, patchSize)
        thresold[i, 0] = var[i, 0] * inverseGammaCDF
        thresold[i, 0] = np.around(thresold[i, 0], decimals=2)
        _, numP = np.shape(patchCollection)
        tempCollection = np.zeros([patchSize * patchSize, numP])
        count = 0
        textureStrength = np.zeros([numP, 1])
        for n in range(numP):
            patch = patchCollection[:, n]
            grad = np.stack((Dh.dot(patch), Dv.dot(patch)), axis=1)
            cov = (grad.T).dot(grad)
            textureStrength[n, 0] = np.matrix.trace(cov)
            if (textureStrength[n, 0] < thresold[i, 0]):
                tempCollection[:, count] = patch
                count = count + 1  # count: number of Low rank patches
        estimatedVariance = var[i, 0]
        # Stoping criteria for stable varince        
        if (count < patchSize * patchSize):
            break
        if (var[i, 0] < 0.1):
            print("Noise free image")
            break
        if (i > 0):
            if (abs(var[i, 0] - var[i - 1, 0]) <= 0.1):
                break
        patchCollection = np.zeros([patchSize * patchSize, count])
        patchCollection[:, 0:] = tempCollection[:, 0:count]
    return estimatedVariance


# save data to excels
def xw_toExcel(data, fileName):
    workbook = xw.Workbook(fileName)  # create excels
    worksheet1 = workbook.add_worksheet("sheet1")  # create sheet
    worksheet1.activate()  # activate
    title = ['image', 'noise_level']  # table title
    worksheet1.write_row('A1', title)
    i = 2
    for j in range(len(data)):
        insertData = [data[j]["id"], data[j]["noiseLevel"]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()  # close excels


xwdata = []
# excel name
excelName = ''
# image path
imgFilePath = ''
for filename in os.listdir(imgFilePath):
    if filename.endswith('jpeg') or filename.endswith('tif'):
        imgPath = imgFilePath + '/' + filename
        grayImage = cv2.imread(imgPath, 2)  # 2-Open as a grayscale map while retaining the in-situ depth

        patchSize = 3
        confidenceLevel = 1 - 1e-6  # choose close to 1
        numIteration = 3

        EV = noiseLevelEstimation(grayImage, patchSize, confidenceLevel, numIteration)
        if EV > 0:
            std = math.sqrt(EV)
        print("img: {}".format(filename), " noise std: {}".format(std))
        xwdata.append({"id": filename, "noiseLevel": std})

xw_toExcel(xwdata, excelName)
