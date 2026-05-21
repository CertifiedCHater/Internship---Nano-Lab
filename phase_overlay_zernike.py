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


# Uses the built-in blank screen function to show a given grayscale value on the full SLM.
# Then we use the Zernike functions as an overlay.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# We need to set the SLM into phase modulation mode to be able to apply Zernike values (instead, we could set a wavelength):
err = HEDS.SDK.libapi.heds_slm_set_modulation(slm.id(), HEDSSLMMOD_Phase)
# err = slm.setWavelength(532.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Configure the blank screen:
grayValue = 128

# Show gray value on SLM without using a handle:
err = slm.showBlankScreen(grayValue)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

zernikeRadius = slm.height_px() / 2.0 + 0.5  # default Zernike radius in HOLOEYE Pattern Generator

# zernikeDataVector consists of:
# index 0: Zernike radius in pixel.
# index 1: tip (blazed grating with deviation ix x-direction).
# index 2: tilt (blazed grating with deviation ix y-direction).
# index 3: second order astigmatism.
# index 4: defocus (r^2), has the same effect like a lens.
# ...
# The vector does not need to hold all elements, it just must have the size up to the last non-zero element, e.g.

# We also can create the vector in more general way and set the Zernike coefficients by their names:
zernikeDataVector = (ctypes.c_float * HEDSZER_COUNT)()
zernikeDataVector[HEDSZER_RadiusPx] = zernikeRadius  # default is half diagonal of SLM in pixel.
zernikeDataVector[HEDSZER_TiltX] = 0.2
zernikeDataVector[HEDSZER_TiltY] = 0.1
zernikeDataVector[HEDSZER_Defocus] = 1.0
zernikeDataVector[HEDSZER_ComaX] = 0.25
print(HEDS.SLM.ZernikePrintString(zernikeDataVector))

# Apply the Zernike paramters to the SLM:
err = slm.zernikeApplyParams(zernikeDataVector)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Now the previously shown blank screen was overlayed with the phase function related to the given Zernike parameters.
# The Zernike overlay is globally applied, i.e. all display functions will be overlayed by the corresponding phase function.

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

