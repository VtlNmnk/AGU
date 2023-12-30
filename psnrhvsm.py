import numpy as np
import cv2
import time

# -------------------------------- [ psnrhvsm ] -------------------------------- #
def psnrhvsm_8x8(imageArray1, imageArray2):
    """
    Calculation of PSNR-HVS-M and PSNR-HVS image quality measures in 8x8 block
    """
    container1 = np.zeros((9, 9), dtype=int)
    container2 = np.zeros((9, 9), dtype=int)
    container1[:8,:8] = imageArray1
    container2[:8,:8] = imageArray2
    p_hvs_m, p_hvs, mse_hvs_m, mse_hvs = psnrhvsm(container1, container2)
    return p_hvs_m, p_hvs, mse_hvs_m, mse_hvs
# -------------------------------- [ psnrhvsm ] -------------------------------- #
def psnrhvsm(imageArray1, imageArray2, wstep = 8):
    """
        Calculation of PSNR-HVS-M and PSNR-HVS image quality measures

        It's a direct implemenation of psnrhvsm.m

        PSNR-HVS-M is Peak Signal to Noise Ratio taking into account
        Contrast Sensitivity Function (CSF) and between-coefficient
        contrast masking of DCT basis functions
        PSNR-HVS is Peak Signal to Noise Ratio taking into account only CSF

        Copyright(c) 2006 Nikolay Ponomarenko

        Python version by Sergey S. Krivenko
        2018/08/27

        Usage:
        import cv2
        import psnrhvsm as p
        imageArray1 = cv2.imread((FileName1), 0)
        imageArray2 = cv2.imread((FileName2), 0)
        psnrhvsm, psnrhvs, mse_hvs_m, mse_hvs = p.psnrhvsm(imageArray1, imageArray2)
    """

    if imageArray1.shape[0] != imageArray2.shape[0] or imageArray1.shape[1] != imageArray2.shape[1]:
        raise ValueError('Images\' dimensions are not equal')
    else:
        LenXY = imageArray1.shape
        LenX = imageArray1.shape[0]
        LenY = imageArray1.shape[1]

    CSFCof = np.array([[1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
            [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
            [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
            [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
            [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
            [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
            [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
            [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]])

    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
            [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
            [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
            [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
            [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
            [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
            [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
            [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])

    step = wstep
    S1 = 0
    S2 = 0
    Num = 0
    X = 0
    Y = 0
    img1 = imageArray1
    img2 = imageArray2
    whileConst = 8

    while Y < LenY - whileConst:
        while X < LenX - whileConst:
            A = np.float64(img1[Y:Y + whileConst, X: X + whileConst])
            B = np.float64(img2[Y:Y + whileConst, X: X + whileConst])
            A_dct = cv2.dct(np.float64(A))
            B_dct = cv2.dct(np.float64(B))
            MaskA = maskeff_optimize(A, A_dct)
            MaskB = maskeff_optimize(B, B_dct)
            if MaskB > MaskA:
                MaskA = MaskB
            X = X + step
            for k in range(8):
                for l in range(8):
                    u = abs(A_dct[k, l] - B_dct[k, l])
                    S2 = S2 + pow(u * CSFCof[k, l], 2)  # PSNR - HVS
                    if k != 0 or l != 0:
                        if u < MaskA / MaskCof[k, l]:
                            u = 0
                        else:
                            u = u - MaskA / MaskCof[k, l]
                    S1 = S1 + pow(u * CSFCof[k, l], 2)  # PSNR - HVS - M
                    Num = Num + 1
        X = 1
        Y = Y + step

    if Num != 0:
        S1 = S1 / Num
        S2 = S2 / Num
        if S1 == 0:
            p_hvs_m = 100000
        else:
            p_hvs_m = 10 * np.log10(255 * 255 / S1)
        if S2 == 0:
            p_hvs = 100000
        else:
            p_hvs = 10 * np.log10(255 * 255 / S2)

    mse_hvs_m = S1
    mse_hvs = S2

    return p_hvs_m, p_hvs, mse_hvs_m, mse_hvs
# -------------------------------- [ psnrhvsm ] -------------------------------- #

# -------------------------------- [ maskeff ] -------------------------------- #
def maskeff(z, zdct):
    """
        Calculation of Enorm value
    """
    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
    [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
    [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
    [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
    [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
    [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
    [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
    [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])
    m = 0
    for k in range(8):
        for l in range(8):
            if k != 0 or l != 0:
                m = m + pow(zdct[k, l], 2) * MaskCof[k, l]

    pop = vari(z)
    if pop != 0:
        pop = (vari(z[0:4, 0:4])+vari(z[0:4, 4:8])+vari(z[4:8, 4:8])+vari(z[4:8, 0:4])) / pop
    m = np.sqrt(m * pop) / 32
    return m
# -------------------------------- [ maskeff ] -------------------------------- #

# -------------------------------- [ maskeff_optimize ] -------------------------------- #
def maskeff_optimize(z, zdct):
    """
        Calculation of Enorm value, speed 2x optimize version
    """
    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
    [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
    [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
    [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
    [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
    [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
    [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
    [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])

    m = np.sum(pow(zdct, 2) * MaskCof) - pow(zdct[0, 0], 2) * MaskCof[0, 0]
    pop = vari(z)
    if pop != 0:
        pop = (vari(z[0:4, 0:4])+vari(z[0:4, 4:8])+vari(z[4:8, 4:8])+vari(z[4:8, 0:4])) / pop
    m = np.sqrt(m * pop) / 32
    return m
# -------------------------------- [ maskeff_optimize ] -------------------------------- #

# -------------------------------- [ vari ] -------------------------------- #
def vari(AA):
    """

    """
    return np.var(AA, ddof=1)*AA.shape[0]*AA.shape[1]
# -------------------------------- [ vari ] -------------------------------- #