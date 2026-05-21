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


# This example requires an NVIDIA Quadro GPU with an NVIDIA Mosaic hardware setup.
# Please see NVIDIA webpage for more information about the Mosaic feature:
# https://www.nvidia.com/en-us/design-visualization/solutions/nvidia-mosaic-technology/
#
# NVIDIA Mosaic can be used with more than two SLMs, but all devices included in the virtual mosaic display need to be of the same type.
#
# By using NVIDIA Mosaic setup, all SLM devices can be operated with perfectly synchronized frame switching using the graphics card hardware synchronization.
#
# In this example, an SLMWindow is initialized, which should be an NVIDIA Mosaic setup.
# Our SLMWindow class supports applying the default layout for the underlying hardware setup.
# In case of the single SLM hardware device, the whole SLM is setup with one SLM screen.
# In case of the NVIDIA Mosaic setup, the SLMWindow area is separated into the underlying hardware SLMs automatically.
# In principle it would be possible with HOLOEYE SLM Display SDK to further split the SLM devices into more virtual SLM screen areas,
# but this is not part of this small example code.
#
# After initialization, a built-in vortex phase function is displayed on each hardware SLM. The SLM index is used for the vortex order.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open the EDID device detection GUI to retrieve the SLMWindow for the whole NVIDIA Mosaic setup:
wnd = HEDS.SLMWindow()
assert wnd.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(wnd.errorCode())

# Set SLM preview scale to "Fit":
err = wnd.preview().setSettings(HEDSSLMPF_None, 0.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Returns the list of hardware SLM devices the SLM window was opened on.
# The devices may be set up in a layout with a number of columns and/or rows.
# Normally the returned layout is 1x1, except if an NVIDIA Mosaic screen was configured and selected for the SLM window:
slms = wnd.slmSetupApplyDefault()
assert wnd.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(wnd.errorCode())

print("SLM display device layout within SLM window:")
print("  Columns: %2d" % wnd.deviceColumns())
print("  Rows:    %2d" % wnd.deviceRows())

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

