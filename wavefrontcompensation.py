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


# This example demonstrates how to apply a wavefront compensation from a *.h5 file provided by HOLOEYE together with the device.
# For demonstration purpose there is a demo-file included in this example, which will not improve the wavefront shape for any
# device, but is clearly visible.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve an SLM window, and open the SLM Preview window in "Fit" mode for the selected SLM:
slmWindow = HEDS.SLMWindow("", True, 0.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Increase SLM preview size, and if available, place it on a secondary monitor:
slmWindow.preview().autoplaceLayoutOnSecondaryMonitor()

# Enable wavefrontcompensation visualization in SLM preview window and stay with SLM preview scale "Fit":
slmWindow.preview().setSettings(HEDSSLMPF_ShowWavefrontCompensation, 0.0)

# Clears the current SLM setup within this SLM window and applies the default SLM layout, which has one SLM screen area per SLM display device:
slms = slmWindow.slmSetupApplyDefault()
assert slmWindow.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmWindow.errorCode())


for slm_index in range (len(slms)):
    slm = slms[slm_index]
    # Give each SLM an incident laser wavelength, so that the wavefront compensation data loaded from file can be converted into phase values:
    err = slm.setWavelength(532.0)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    # Show a different blank screen on each SLM, start with 128 for the first SLM:
    err = slm.showBlankScreen(int(128+64*slm_index))
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Retrieve the SLM device layout within slmWindow:
dev_cols = slmWindow.deviceColumns()
dev_rows = slmWindow.deviceRows()

# Apply a wavefront compensation file on each SLM display device available within slmWindow:
for dev_row in range(dev_rows):
    for dev_col in range(dev_cols):
        # Show the blank screen for some time before loading the wavefront compensation file:
        print("Blank screen with gray level 128.")
        if not slmWindow.wait(2500) == HEDSERR_NoError:   exit(-1)

        # Load a demo wavefront compensation file (*.h5):
        wavefrontCompensationFilename = "data/wfcdemo_holoeye_logo.h5"
        err = slmWindow.loadWavefrontCompensationFile(wavefrontCompensationFilename, dev_col, dev_row)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        # Wait a little before applying the wavefront compensation to the next device:
        print("Wavefront compensation is now applied on SLM display device at column=", dev_col, " and row=", dev_row, ".")
        if not slmWindow.wait(1000) == HEDSERR_NoError:   exit(-1)

# Show all the compensated blank screens for some time:
print("Wavefront compensation is now applied to all SLMs. Waiting for 5 seconds to continue ...")
if not slmWindow.wait(5000) == HEDSERR_NoError:   exit(-1)

# Clear all previously loaded wavefront compensations:
for dev_row in range( 0, dev_rows):
    for dev_col in range( 0, dev_cols):
        err = slmWindow.clearWavefrontCompensation(dev_col, dev_row)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

print("Wavefront compensation is now cleared on all SLMs.")

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

