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


# Calculation of monochrome image data with float 32bit type.
# Calculation for one line only, use of show flag PresentAutomatic for tiling into full SLM screen.
# Image content is gray value ramp with a configurable width, calculated using float data type for a single line in full SLM width.
# The gray values of the ramp have a range from 0 to 1, non-fitting values will be cropped to either 0 (black) or 1 (white).


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())


# Configure the ramp width:
rampWidth = slm.width_px() / 2

# Configure the calculated image size in pixel:
dataWidth = slm.width_px()
dataHeight = 1

# Create an image pixel data field in memory to be shown on an SLM. Each pixel contains a float gray value:
data = HEDS.SLMDataField(dataWidth, dataHeight, HEDSDTFMT_FLOAT_32)

# Calculate the data:
for i in range(0, dataWidth):
    x = float(i - dataWidth / 2)
    rampScaleFactor = float(dataWidth) / float(rampWidth)
    err = data.setPixel(i, 0, float(0.5 + x / float(dataWidth) * rampScaleFactor) )
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

print(data.printString("data"))

# Show the pixel data field on the SLM (the single line will be spread to fullscreen using show flag HEDSSHF_PresentAutomatic):
err = slm.showImageData(data)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

