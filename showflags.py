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


# This example load an RGB image file to demonstrate different show flag effects.
# Note: All show flags can technically be set simultaneously, but if multiple present
# show flags are set, only one of can be active. It is undefined behavior if multiple
# present show flags are set.
# HEDSSHF_PresentAutomatic is always set when no other present show flags are set.
#
# For phase data, it is recommended to use present show flags automatic or tiling,
# which do pixel-exact addressing. If phase data shall be presented with a different
# scale, we recommend to recalculate the phase data with the new properties.


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

# Default: showflags = HEDSSHF_PresentAutomatic:
# Two-dimensional data is shown with HEDSSHF_PresentCentered, assuming it is any kind of image or phase data.
# One-dimensional data is shown with HEDSSHF_PresentTiledCentered, assuming it is a repetitive optical phase function.
# This is recommended for phase and image data.
err = dataHandle.show(HEDSSHF_PresentAutomatic)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_PresentAutomatic)")

 # showflags = HEDSSHF_PresentCentered:
 # The data is shown unscaled at the center of the SLM. Free areas are filled with black (0).
 # If the data is larger than the SLM, it is cropped to the SLM size.
 # This is recommended for image data.
err = dataHandle.show(HEDSSHF_PresentCentered)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_PresentCentered)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_PresentFitWithBars:
# Resizes the data to fully fit into the SLM. Keeps the aspect ratio.
# Free areas on the top/bottom or left/right of the SLM will be filled with black (0).
# This is recommended for image data.
err = dataHandle.show(HEDSSHF_PresentFitWithBars)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_PresentFitWithBars)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_PresentFitNoBars:
# Resizes the data to fully fill the SLM. Keeps the aspect ratio.
# Areas on the top/bottom or left/right of the data may be cropped and not visible.
err = dataHandle.show(HEDSSHF_PresentFitNoBars)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_PresentFitNoBars)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_PresentFitScreen:
# Resizes the data to fully fill the SLM. Does not keep the aspect ratio so that no data is cropped.
err = dataHandle.show(HEDSSHF_PresentFitScreen)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_PresentFitScreen)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_PresentTiledCentered:
# Tiles the data
# The data (i.e. the tile) is shown like when using HEDSSHF_PresentCentered, but when HEDSSHF_PresentCentered would
# produce black bars on the edges of the SLM area, HEDSSHF_PresentTiledCentered will replicate the data (i.e. the tile)
# into these black areas to fill the whole SLMs area.
# This is recommended for most types phase data. May not be applicable for all phase data.
err = dataHandle.show(HEDSSHF_PresentTiledCentered)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_PresentTiledCentered)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_TransposeData:
# If set, rows and columns of the data will be switched.
# Not recommended any more. Please use the load flag HEDSLDF_TransposeData instead if your data is stored in Fortran style.
# Our APIs typically do the transpose automatically when necessary.
err = dataHandle.show(HEDSSHF_TransposeData)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_TransposeData)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_FlipHorizontal:
# Flip the data horizontally, i.e. in x direction.
err = dataHandle.show(HEDSSHF_FlipHorizontal)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_FlipHorizontal)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_FlipVertical:
# Flip the data vertically, i.e. in y direction:
err = dataHandle.show(HEDSSHF_FlipVertical)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_FlipVertical)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

# showflags = HEDSSHF_InvertValues:
# Invert the gray levels of the data per color channel (0 -> 255; 255 -> 0; 128 -> 127; 127 -> 128).
err = dataHandle.show(HEDSSHF_InvertValues)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print("dataHandle.show(HEDSSHF_InvertValues)")
assert slm.wait(2500) == HEDSERR_NoError, print("Wait failed!")

print("Finished.\n")

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

