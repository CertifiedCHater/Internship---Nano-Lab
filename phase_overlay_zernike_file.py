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


# Shows a blank screen with gray level 128 and sets a Zernike overlay loaded from a Zernike
# parameter definition file, e.g. saved by HOLOEYE SLM Pattern Generator software.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm  = HEDS.SLM.Init("", True, 0.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# We need to set the SLM into phase modulation mode to be able to apply Zernike values (instead, we could set a wavelength):
err = HEDS.SDK.libapi.heds_slm_set_modulation(slm.id(), HEDSSLMMOD_Phase)
# err = slm.setWavelength(532.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Set the additional flag HEDSSLMPF_ShowZernikeRadius to programatically press the button to
# enable the Zernike radius visualization in SLM preview windows realtime preview,
# and leave SLM preview scale in "Fit mode:
err = slm.preview().setSettings(HEDSSLMPF_ShowZernikeRadius, 0.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Configure the blank screen:
grayValue = 128

# Show gray value on SLM without using a handle:
err = slm.showBlankScreen(grayValue)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Load the Zernike paramter values from file:
err, zernikeParameters = slm.zernikeLoadParamsFromFile("data/zernike_parameters.zernike.txt")
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print(HEDS.SLM.ZernikePrintString(zernikeParameters, "Loaded Zernike parameters from file"))

# Wait 3 seconds before applying loaded Zernike parameters:
print("\nWaiting 3 seconds before applying Zernike parameters ...")
slm.wait(3000)

# Apply the loaded Zernike paramters to the SLM:
err = slm.zernikeApplyParams(zernikeParameters)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

print("Applied Zernike parameters to SLM.")

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

