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


# A color gradient is displayed, e.g. using sine curves with period = SLM width,
# each color shifted in phase by 120 degrees.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

import math


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Configure the calculated image size in pixel:
dataWidth = slm.width_px()
dataHeight = 1

# Create an image pixel data field in memory to be shown on an SLM. Each pixel contains an RGB tuple:
data = HEDS.SLMDataField(dataWidth, dataHeight, HEDSDTFMT_INT_RGB24,  HEDSSHF_PresentFitScreen, 0 ,False)

# Parameter settings:
# Color phase Shift is 120 degree per color channel.
# Color value offset is needed to avoid negativ values.
phaseShiftColor = float(2.0 * math.pi / 3.0)
offset = 1.0

# Calculate the data:
for x in range(0, dataWidth):
    r = int((math.sin(float(x) / dataWidth * 2.0*math.pi + phaseShiftColor) + offset) / 2.0 * 255.0)
    g = int((math.sin(float(x) / dataWidth * 2.0*math.pi                  ) + offset) / 2.0 * 255.0)
    b = int((math.sin(float(x) / dataWidth * 2.0*math.pi - phaseShiftColor) + offset) / 2.0 * 255.0)
    err = data.setPixel(x, 0, HEDS.heds_rgb24(r,g,b))
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

print(data.printString("data"))

# Show the image pixel data field on the SLM (the single line will be spread to fullscreen using show flag HEDSSHF_PresentFitScreen):
err = slm.showImageData(data)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

