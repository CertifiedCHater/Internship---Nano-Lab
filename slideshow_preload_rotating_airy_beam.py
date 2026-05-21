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


# Calculates and pre-loads airy beam phase functions with different rotating angles
# on an SLM in a slideshow. Due to the pre-calculation, this example is expected
# to run as fast as possible after the phase data fields are uploaded.
# The uploaded data can be played repeatedly forever without any additional calculation.
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

import concurrent.futures


# A helper function to be able to run multiple phase data field computations in parallel.
def computeAiryBeamThreadFunc(rotAngleDeg):
    # Compute the data using the imported function:
    if not HEDS.supportNumPy:
        phaseDataField = computeAiryBeam(dataWidth, dataHeight, centerX, centerY, innerRadius, rotAngleDeg, onedimensional)
    else:
        phaseDataField = computeAiryBeamNumPy(dataWidth, dataHeight, centerX, centerY, innerRadius, rotAngleDeg, onedimensional)

    # Load the data and retrieve the data handle:
    err, datahandle = slm.loadPhaseData(phaseDataField)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    assert datahandle.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(datahandle.errorCode())

    # Give the data a number of video frames to be visible on each show event later:
    err = datahandle.setDuration(visibleDurationInFrames)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    print(".", end='')
    return datahandle, rotAngleDeg


if __name__ == "__main__":
    # Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
    err = HEDS.SDK.Init(4,0)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    # Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
    slm = HEDS.SLM.Init("", True, 0.0)
    assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

    # Configure this example:
    dataWidth = slm.width_px()
    dataHeight = slm.height_px()
    numberOfAngleSteps = int(360/3)  # please reduce in case your PC cannot handle this.
    visibleDurationInFrames = 1      # please increase to play slower.
    repeatSlideshow = 5              # 0 will play forever.

    # Configure the airy beam properties:
    onedimensional = False
    innerRadius = min(dataWidth, dataHeight) / 10.0
    centerX = 0
    centerY = 0

    # Pre-compute airy beam functions for different rotation angles and load them into SLM Display SDK:
    txtUsing = "using slower for-loop (ctypes array) implementation"
    if HEDS.supportNumPy:
       txtUsing = ("using numpy matrix multiplication implementation")

    print("Pre-computing airy beam data fields (", txtUsing, "):")
    datahandle_list = []
    startTimeCalc = HEDS.SDK.libapi.heds_time_now()

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        for steps in range(0, numberOfAngleSteps):
            rotAngleDeg = 360.0 / numberOfAngleSteps * steps
            futures.append(executor.submit(computeAiryBeamThreadFunc, rotAngleDeg))

        for future in concurrent.futures.as_completed(futures):
            datahandle_list.append(future.result())

    # Sort list of data handles by rotation angle after all threads have messed up the order:
    datahandle_list.sort(key=lambda tup: tup[1])

    endTimeCalc = HEDS.SDK.libapi.heds_time_now()
    calcDuration = HEDS.SDK.libapi.heds_time_duration_ms(endTimeCalc, startTimeCalc)
    print("\nCalculating phase fields took " + "%.1f" % (calcDuration/1000.0) + " seconds.")

    # Play the pre-loaded data in a slideshow multiple times without any image or phase data computation:
    r = 0  # count runs of the slideshow.
    while (r < repeatSlideshow) or (repeatSlideshow <= 0):
        r += 1
        print("Show slideshow for the ", r, " time ... ", end='')

        startTimeSlideshow = HEDS.SDK.libapi.heds_time_now()

        # Run a slideshow over all uploaded data handles:
        for datahandle, angle in datahandle_list:
            err = datahandle.show()
            assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        endTimeSlideshow = HEDS.SDK.libapi.heds_time_now()
        slideshowDuration = HEDS.SDK.libapi.heds_time_duration_ms(endTimeSlideshow, startTimeSlideshow)
        avgFrameTime = slideshowDuration / float(len(datahandle_list))
        print(" <-- Running for %.1f seconds" % (slideshowDuration / 1000.0) + " | Avg. frame time: %7.2fms" % avgFrameTime + " | FPS: %.1f" % (1000.0/avgFrameTime))

    print("Slideshow finished.")

    # Wait until each opened SLM window was closed manually by using the tray icon GUI:
    HEDS.SDK.WaitAllClosed()


