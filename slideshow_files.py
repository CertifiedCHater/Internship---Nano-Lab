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


# This example loads a list of image files into the video memory of the GPU and then
# shows the data fluently by telling HOLOEYE SLM Display SDK the next data handle to show.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

import sys, os


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Specify the folder with the slideshow to be played. Use a default folder in case no command line parameters are passed:
thisScriptPath = os.path.dirname(__file__)
imageFolder = os.path.join(thisScriptPath, "data", "vertical_grating")

# Please select the duration in ms each image file shall be shown on the SLM
dataDisplayDurationInVideoFrames = 1

# Configure how often the slideshow should play:
repeatSlideshow = 10  # 0 will play forever.

# Please select how to scale and transform image files while displaying:
showflags = HEDSSHF_PresentAutomatic

# Overwrite the default path if the first command line parameter is available:
if len(sys.argv) > 1:
    imageFolder = sys.argv[1]

# Search image files in given folder:
print("Loading files from \"" + imageFolder + "\" ...")
filesList = os.listdir(imageFolder)
# Filter *.png, *.bmp, *.gif, *.jpg, and *.jpeg files:
foundFiles = [filename for filename in filesList if str(filename).lower().endswith(".png") or str(filename).lower().endswith(".gif") or str(filename).lower().endswith(".bmp") or str(filename).lower().endswith(".jpg") or str(filename).lower().endswith(".jpeg")]
foundFiles.sort()
print(foundFiles)
print("Number of images found in imageFolder = " + str(len(foundFiles)))

if len(foundFiles) <= 0:
    print("No image files found in given folder \"" + imageFolder + "\".")
    sys.exit(1)


# Open device detection and retrieve one SLM, and open the SLM Preview window with default scale (1:1) for the selected SLM:
slm  = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

print("Expected frame time is " + str((dataDisplayDurationInVideoFrames / slm.refreshrate_hz() * 1000.0)) + " ms.")

# Load the image files into SLM Display SDK:
dataHandleList = []
for file in foundFiles:
    # Files only contain the names, add the path:
    filename = os.path.join(imageFolder, file)

    print("Loading file \"" + filename + "\"")

    # Load the filename into the SDK so that the SDK can load the data from the file:
    err, dataHandle= slm.loadImageDataFromFile(filename)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    assert dataHandle.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(dataHandle.errorCode())

    # Give the data a number of video frames to be visible on each show event later:
    err = dataHandle.setDuration(dataDisplayDurationInVideoFrames)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    # Save the prepared data handles for showing them later:
    dataHandleList.append(dataHandle)


# Play the pre-loaded data in a slideshow multiple times without any image or phase data computation:
r = 0  # count runs of the slideshow.
lastDataHandle = None
currentTime = HEDS.SDK.libapi.heds_time_now()
startTime = currentTime
while (r < repeatSlideshow) or (repeatSlideshow <= 0):
    r += 1

    print("Show slideshow for the "+str(r)+". time ...")

    for dataHandle in dataHandleList:
        err = dataHandle.show(showflags)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        if lastDataHandle is not None:
            currentTime = HEDS.SDK.libapi.heds_time_now()
            dataDuration = HEDS.SDK.libapi.heds_time_duration_ms(currentTime, startTime)
            startTime = currentTime
            print(lastDataHandle.getTimingPrintString() + " | Total data duration: %0.2f" % dataDuration + "ms")

        lastDataHandle = dataHandle

print("Slideshow finished.")

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

