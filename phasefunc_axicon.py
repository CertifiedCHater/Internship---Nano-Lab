# -*- coding: utf-8 -*-

#------------------------- \cond COPYRIGHT --------------------------#
#                                                                    #
# Copyright (C) 2024 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#----------------------------- \endcond -----------------------------#


# Calculates and shows an axicon phase function on an SLM.
# The pixel phase data is calculated on the CPU using 32-bit
# floating point phase values and is addressed using the show
# phase data API function of HOLOEYE SLM Display SDK.
# Phase wrapping into the gray levels of the SLM is done by
# SLM Display SDK internally.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

# This example requires numpy to be installed in your Python interpreter:
import numpy as np
import math


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4, 0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Get the SLM width in number of pixel:
w = slm.width_px()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Get the SLM height in number of pixel:
h = slm.height_px()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Configure the axicon properties:
innerRadius = h / 3  # the radius at which the phase function changed by 2*pi radian.
centerX = 0
centerY = 0

# Calculate the phase values of an axicon in a pixel - wise  matrix:
# pre - calc.helper variables:
phaseModulation = 2.0 * math.pi
dataWidth = w
dataHeight = h

# Calculate a phase data field (of size dataWidth x dataHeight) with the axicon phase function using numpy matrix multiplication:
x = np.linspace(1, dataWidth, dataWidth, dtype=np.float32) - np.float32(dataWidth / 2) - np.float32(centerX)
y = np.linspace(1, dataHeight, dataHeight, dtype=np.float32) - np.float32(dataHeight / 2) - np.float32(centerY)

x2 = np.matrix(x * x)
y2 = np.matrix(y * y).transpose()

phaseData = np.float32(phaseModulation) * np.sqrt( np.array((np.dot(np.ones([dataHeight, 1], np.float32), x2) + np.dot(y2, np.ones([1, dataWidth], np.float32))), dtype=np.float32), dtype=np.float32) / np.float32(innerRadius)

# Show phase data on the SLM:
err = slm.showPhaseData(phaseData)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()
