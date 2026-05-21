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


# This example loads a small version of the RGB image, and then demonstrates how to apply different
# transform options through the data handle, without the need to upload any new image data any more.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm  = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Load image pixel data from a file on disk into a data handle of SLM Display SDK:
filename = "data/RGBCMY01_640x480.png"
err, dataHandle = slm.loadImageDataFromFile(filename)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Show the image with default showflags (HEDSSHF_PresentAutomatic), because other present show flags may disable the transform feature below:
err = dataHandle.show()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show()")

err = dataHandle.setTransformScale(1.5)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformScale(1.5)")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformScale(1.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformScale(1.0)")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformScale(0.5)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformScale(0.5)")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformScale(0.1)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformScale(0.1)")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformScale(10.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformScale(10.0)")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformScale(0.7)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformScale(0.7)")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformShift((slm.width_px()/4), (slm.height_px ()/4))
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformShift((slm.width_px()/4), (slm.height_px()/4))")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformShift((-slm.width_px()/4), (-slm.height_px()/4))
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformShift((-slm.width_px()/4), (-slm.height_px()/4))")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

err = dataHandle.setTransformShift(0, 0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err = dataHandle.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.setTransformShift(0, 0)")
if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

