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


# Simple example of loading an image file containing RGB image data.
# The data coud have been loaded using SLM Display SDK (see example load_image),
# but here we want to demonstrate how to load the data manually from file using
# this programming language, and then load the data into the video memory of
# HOLOEYE SLM Display SDK, instead of only providing a filename.
# This is useful if you want to load data from an image file and make changes
# to the data before passing into the SDK.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

# This example requires numpy and PIL to be installed in your Python interpreter:
import numpy as np
from PIL import Image


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm  = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Load image data from a file on disk manually:
img = Image.open("data/RGBCMY01.png")
if img is None: exit(1)

# Convert image into Numpy array:
imageData = np.asarray(img)

# Load image data into video memory of SLM Display SDK:
err, dataHandle = slm.loadImageData(imageData)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Show the returned data handle on the SLM:
err = dataHandle.show()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

