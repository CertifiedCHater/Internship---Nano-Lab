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


# Calculates and shows airy beam phase functions with different rotating angles
# on an SLM in a slideshow. Due to the live calculation, this example is expected
# to run slower than the preload version of this example.
# The pixel phase data is calculated on the CPU using 32-bit
# floating point phase values and is addressed using the show
# phase data API function of HOLOEYE SLM Display SDK.
# Phase wrapping into the gray levels of the SLM is done by
# SLM Display SDK internally.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

# This example does not require numpy to be installed, but installing numpy improves computation performance:
if HEDS.supportNumPy:
    # Import helper function to compute the phase data field of the airy beams,
    # using the numpy matrix multiplication implementation:
    from HEDS.functions import computeAiryBeamNumPy
else:
    # Import helper function to compute the phase data field of the airy beams,
    # using the ctypes array based implementation:
    from HEDS.functions import computeAiryBeam

    print("Warning: Image calculation is done without using numpy.\n"
          "         Please install numpy package to improve computation performance.")


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Configure the airy beam properties:
dataWidth = slm.width_px()
dataHeight = slm.height_px()
innerRadius = min(dataWidth, dataHeight) / 10.0
centerX = 0
centerY = 0
onedimensional = False

# Configure the slideshow:
angleDegStart = 0.0
angleDegStep = 1.0
angleDegEnd = 360.0

# Pre-compute airy beam functions for different rotation angles and load them into SLM Display SDK:
txtUsing = "using slower for-loop"
if HEDS.supportNumPy:
   txtUsing = ("using numpy")

print("Live computing airy beam data fields (", txtUsing, ").")

startTime = HEDS.SDK.libapi.heds_time_now()
endTimeCalc = startTime

angleDeg = angleDegStart
while angleDeg < angleDegEnd:
    if HEDS.supportNumPy:
        phaseDataField = computeAiryBeamNumPy(dataWidth, dataHeight, centerX, centerY, innerRadius, angleDeg, onedimensional)
    else:
        phaseDataField = computeAiryBeam(dataWidth, dataHeight, centerX, centerY, innerRadius, angleDeg, onedimensional)

    endTimeCalc = HEDS.SDK.libapi.heds_time_now()

    err = slm.showPhaseData(phaseDataField)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    endTimeTotal = HEDS.SDK.libapi.heds_time_now()
    calcTime = HEDS.SDK.libapi.heds_time_duration_ms(endTimeCalc, startTime)
    totalFrameTime = HEDS.SDK.libapi.heds_time_duration_ms(endTimeTotal, startTime)

    print("angleDeg = %8.1f" % angleDeg + " | calcTime = %4.1fms" % calcTime + " | totalFrameTime = %4.1fms" % totalFrameTime)

    # Restart the time measurement for next frame:
    startTime = HEDS.SDK.libapi.heds_time_now()

    angleDeg += angleDegStep

print("Slideshow finished.")

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()


