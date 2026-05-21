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


# A single SLMWindow is opened on a single SLM hardware device, and the SLMWindow is separated into two independent virtual SLMs.
# After initialization, a vortex phase function is displayed on each virtual SLM area by using the built-in vortex function.
# The vortex order is defined by the index of the SLM for visualization.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve an SLMWindow:
wnd = HEDS.SLMWindow()
assert wnd.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(wnd.errorCode())

# Set SLM preview scale to "Fit":
wnd.preview().setSettings(HEDSSLMPF_None, 0.0)

# Add an SLM into the SLM setup of SLMWindow by providing a geometry within the SLMWindow for the added SLM:
err = wnd.slmSetupAdd(HEDS.RectGeometry(0, 0, wnd.width_px() / 2, wnd.height_px()))
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Add another SLM into the SLM setup of SLMWindow by providing a second geometry within the SLMWindow for the added SLM:
# If both added SLMs would overlap or the geometry does not fit into the SLMWindow, an error is returned here.
err = wnd.slmSetupAdd(HEDS.RectGeometry(wnd.width_px() / 2, 0, wnd.width_px() / 2, wnd.height_px()))
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Applies the SLM setup of the SLMWindow and returns the objects for all SLM within the SLMWindow:
slms = wnd.slmSetupApply()

# Show some data on each SLM screen:
dataHandles = []
for index in range(0, len(slms)):
    # Preload a vortex phase function for each SLM using the built-in function:
    vortex_order = index + 1
    err, dataHandle_id = slms[index].loadVortex(vortex_order)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    # Store the returned data handle in datahandle_list to show them all together later:
    dataHandles.append(dataHandle_id)

# Show all data handles simultaneously on their SLM:
err = HEDS.SDK.ShowDataHandles(dataHandles)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

