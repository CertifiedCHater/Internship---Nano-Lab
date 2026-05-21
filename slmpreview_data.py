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


# An example to demonstrate how to retrieve the currently addressed image or phase data
# from the SLM screen, without the need to show the SLM preview window.
# The example addresses a color image, captures two images, one in color format and one in monochrome format.
# Capturing in monochrome 8-bit format is often sufficient when working with phase modulating SLMs.
# Anyway, the captured image always contains color data and is internally converted into a grayscale image
# when retrieving monochrome image data.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

# This example requires numpy and PIL to be installed in your Python interpreter:
import numpy as np
from PIL import Image


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and do not open the SLM Preview window:
slm = HEDS.SLM.Init("", False)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Show some colored data on the SLM:
filename = "data/RGBCMY01.png"
err = slm.showImageDataFromFile(filename)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Wait a little to make sure the shown data is already visible and can be captured:
slm.wait(int(slm.frametime_ms()))


# Retrieve the preview image data into numpy arrays for capturing with and without color format:
err, image_data_mono = HEDS.SDK.libapi.heds_slmwindow_get_image_data(slm.id().slmwindow_id, HEDSDTFMT_INT_U8)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
err, image_data_rgb = HEDS.SDK.libapi.heds_slmwindow_get_image_data(slm.id().slmwindow_id, HEDSDTFMT_INT_RGB24)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Give some info about the captured images on command line:
print("image_data_mono.shape = " + str(image_data_mono.shape))
print("image_data_rgb.shape = " + str(image_data_rgb.shape))

img_mono = Image.fromarray(image_data_mono, "L")
img_rgb = Image.fromarray(image_data_rgb, "RGB")

img_mono.save("data/captured_image_mono.png")
img_rgb.save("data/captured_image_rgb.png")

# Show the image in a GUI widget:
img_mono.show()
img_rgb.show()

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

