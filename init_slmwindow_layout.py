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


# An SLMWindow is initialized and several SLMs are created using a layout within the SLMWindow,
# i.e. the hardware SLM screen is divided into multiple virtual SLMs.
# Creates a 4 by 3 matrix of SLMs. After initialization, a vortex phase function is addressed
# on each virtual SLM. The SLM index is used for the vortex order for visualization of the SLM index.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLMWindow, and open the SLM Preview window in "Fit" mode for the selected SLM:
wnd = HEDS.SLMWindow("",True, 0.0)
assert wnd.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(wnd.errorCode())

# Applies a setup of SLMs into the SLMWindow and retrieves the list of SLM objects for later usage:
cols = 4
rows = 3
slms = wnd.slmSetupApplyLayout(cols, rows)
assert wnd.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(wnd.errorCode())

# Generate vortex (data handels) for each SLM:
dataHandle_list = []
vortex_order = 1
for i in range(0, len(slms)):
    # Loads data to show an optical vortex phase function:
    vortex_order = i + 1
    # Loads an optical vortex phase function into video memory:
    err, dataHandle = slms[i].loadVortex(vortex_order)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    # Add the handle to the handle list:
    dataHandle_list.append(dataHandle)

# Show all uploaded vortex phase data on their SLM area simultaneously:
err = HEDS.SDK.ShowDataHandles(dataHandle_list)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

