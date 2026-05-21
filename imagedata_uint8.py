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


# Calculation of monochrome image data with unsinged 8bit integer type.
# Calculation only for one line (256 pixel wide), use of show flag HEDSSHF_PresentFitScreen for scaling into full screen.
# In the middle of the image the gray value is 128.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Configure the calculated image size in pixel:
dataWidth = 256
dataHeight = 1

# Create an image pixel data field in memory to be shown on an SLM. Each pixel contains an 8-bit integer gray value:
data = HEDS.SLMDataField(dataWidth, dataHeight, HEDSDTFMT_INT_U8, HEDSSHF_PresentFitScreen)

# Calculate the data:
for x in range(0, dataWidth):
    err = data.setPixel(x, 0, x)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

print(data.printString("data"))

# Show the pixel data field on the SLM (the single line will be spread to fullscreen using show flag HEDSSHF_PresentFitScreen):
err = slm.showImageData(data)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

