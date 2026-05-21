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
# Then we use the beam manipulation provided through the data handles to apply a phase overlay (tip/tilt/lens/offset).


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window with default scale (1:1) for the selected SLM:
slm = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Configure the blank screen where overlay is applied to:
grayValue = 128
grayValueOffset = -128 # this value is applied to the data handle later.

# Configure beam manipulation in physical units:
wavelength_nm = 633.0   # wavelength of incident laser light

steering_angle_x_deg = 0.2
steering_angle_y_deg = -0.3
focal_length_mm = 200.0

# Set a wait duration to allow applying both, beam manipulation and gray value offset within the visible duration for demonstration purpose:
wait_duration_ms = int(float(HEDS_DATAHANDLE_MAX_DURATION * slm.frametime_ms()) / 4.0)

# Apply wavelength to SLM to be able to convert physical units into generalized units for beam manipulation.
# This also enables phase modulation for the SLM, which is necessary to apply beam manipulation.
# In amplitude modulation, beam manipulation values would be ignored:
err = slm.setWavelength(wavelength_nm)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Create a datafield to be uploaded into the SLM. The datafield just consists of a single pixel with the grayValue and will
# automatically be extended into full SLM screen due to HEDSSHF_PresetAutomatic show flag:
data = HEDS.SLMDataField(1, 1)
data.setPixel(0, 0, grayValue)

# Upload the data field into the SLM. This will create an SLMDataHandle (dh), which refers to the data:
err, dh = slm.loadPhaseData(data)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
assert dh.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(dh.errorCode())

# Set the maximum visible duration to demonstrate applying changes to the handle while the handle is within its visible duration:
err = dh.setDuration(HEDS_DATAHANDLE_MAX_DURATION)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Make the data visible on the SLM screen without beam manipulation overlay:
err = dh.show(HEDSSHF_PresentAutomatic)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

print("The data handle of the blank screen with gray value %d is now made visible." %grayValue)

# Wait until we apply the beam manipulation to make the uploaded gray value data visible first:
print("waiting for %d ms." %wait_duration_ms)
slm.wait(wait_duration_ms)

# Calculate proper parameters to pass to the data handle.
# The handle parameters are related to SLM properties and are scaled to produce
# meaningful results in the range between -1.0f and 1.0f. The class BeamManipulation
# allows us to use the heds_utils_beam_xxx functions to convert between the physical
# observables (deviation angles and focal length of the lens) and these generalized
# parameters:
bm = HEDS.BeamManipulation( 0.0, 0.0, 0.0, 0, 0.0, 0.0, None, dh)
bm.setBeamSteerXDegree(steering_angle_x_deg)
bm.setBeamSteerYDegree(steering_angle_y_deg)
bm.setBeamLensFocalLengthMM(focal_length_mm)

print("bm.getBeamSteerX() = %f ==> steering angle x = %f deg" %(bm.getBeamSteerX(), bm.getBeamSteerXDegree()))
print("bm.getBeamSteerY() = %f ==> steering angle y = %f deg" %(bm.getBeamSteerY(), bm.getBeamSteerYDegree()))
print("bm.getBeamLens()   = %f ==> f = %f mm" %(bm.getBeamLens(), bm.getBeamLensFocalLengthMM()))

# Now, after calculating the generalized beam manipulation parameters, we can apply beam manipulation to the data handle,
# and the result will be visible within the next video frame:
err = dh.setBeamManipulation(bm)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Now the new beam manipulation can be made active on the already visible data:
err = dh.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until we apply the gray value offset to make the beam manipulation visible first:
print("waiting for %d ms." %wait_duration_ms)
slm.wait(wait_duration_ms)

print("Applying value offset %d" %grayValueOffset )

# Similarily, we can just apply a gray value offset in all supported data formats (unsigned char, HEDS::RGB24, HEDS::RGBA32, float, double),
# and the offset will be applied instantly, too:
err = dh.setValueOffset(grayValueOffset)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Now the new offset can be made active on the already visible data:
err = dh.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Show a new data handle, which does not have the beam manipulations applied.
# This call waits until the visible duration of the first data handle has ended:
err = slm.showBlankScreen(grayValue)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

print("Next data handle (gray value %d) is now visible, after first data handle has reached its visible duration of %d frames." %(grayValue, HEDS_DATAHANDLE_MAX_DURATION))

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

