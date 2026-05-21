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


# Show monochrome phase data calculated using 32-bit float values.
# Calculates phase values for a blazed grating with a period of 77 pixel.
# We only need to calculate one period for the grating on the CPU, since the
# grating is a repetition of the period. When the grating is aligned to the
# pixel of the SLM, HOLOEYE SLM Display SDK can do the replication very fast
# on the GPU during rendering. The default show flag HEDSSHF_PresentAutomatic
# would figure out that the data has a size of only one pixel in one direction
# and would treat it as a grating anyway, but for demonstration purpose we pass
# the show flag HEDSSHF_PresentTiledCentered manually, which enforces the
# replication during rendering.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

import math


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window with default scale (1:1) for the selected SLM:
slm = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Configuration params:
blazePeriod = 77
phase_unit = 2.0 * math.pi

# Set the calculated data size (calculate single period for speed-up):
dataWidth = blazePeriod
dataHeight = 1

# Pixel data field for the blazed grating period using 32-bit floating point values:
phaseData = HEDS.SLMDataField(dataWidth, dataHeight, HEDSDTFMT_FLOAT_32, HEDSSHF_PresentTiledCentered, 0, False)

# Calculate the data:
for x in range(0, dataWidth-1):
    phaseData.setPixel(x, 0, float(x * phase_unit) / float(blazePeriod))

# Show the phase data on the SLM:
err = slm.showPhaseData(data=phaseData, flags=None, phase_unit=phase_unit)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

