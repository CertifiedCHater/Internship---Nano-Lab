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


# This example just opens one SLM on a single SLMWindow and shows all different kind of built-in show routines in a slideshow.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *


# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window for the selected SLM.
# Set the SLM preview scale to 0.0 for "Fit" mode:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Move SLM preview on a secondary monitor, if available, to enlarge it:
slm.preview().autoplaceLayoutOnSecondaryMonitor()

# Configure slideshow:
repeatSlideshow = 3     # 0: forever, until SLM is closed through tray icon.

# Run the slideshow:
r = 0   # Count repeats of the slideshow.
while ((r < repeatSlideshow) or (repeatSlideshow <= 0)) :
    r += 1

    print("\nRepeating built-in slideshow for the " + str(r) + ". time.")

    # Show a blank screen from an 8-bit gray value:
    err = slm.showBlankScreen(int(128))
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showBlankScreen(int(128))")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a blank screen from a 32-bit floating point gray value (cropped outside of 0.0f and 1.0f):
    err = slm.showBlankScreen(0.25)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showBlankScreen(0.25)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a blank screen from a 64-bit floating point gray value (cropped outside of 0.0 and 1.0):
    err = slm.showBlankScreen(0.75)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showBlankScreen(0.75)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a horizontal binary grating with a period of 2*100 pixel, from 8-bit gray values 0 and 128:
    err = slm.showBlankScreen(heds_rgb24(0, 128, 0))
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showBlankScreen(heds_rgb24(0, 128, 0))")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a blank screen from a 32-bit RGBA blue color value (alpha channel is ignored internally):
    err = slm.showBlankScreen(heds_rgba32(64, 128, 192, 255))
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showBlankScreen(heds_rgba32(64, 128, 192, 255))")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a horizontally divided screen from two different 64-bit floating point gray values:
    err = slm.showDividedScreenHorizontal(0.0, 0.5)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showDividedScreenHorizontal(0.0, 0.5)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a vertically divided screen from two different 64-bit floating point gray values:
    err = slm.showDividedScreenVertical(0.0, 0.5)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showDividedScreenVertical(0.0, 0.5)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a horizontal binary grating with a period of 2*100 pixel, from 8-bit gray values 0 and 128:
    err = slm.showGratingBinaryHorizontal(int(100), int (100), int(0), int(128), 0)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showGratingBinaryHorizontal(int(100), int (100), int(0), int(128), 0)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a verical binary grating with a period of 2*100 pixel, from 64-bit floating point gray values 0.0 and 0.5:
    err = slm.showGratingBinaryVertical(100, 100, 0.0, 0.5)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showGratingBinaryVertical(100, 100, 0.0, 0.5)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a horizontal blazed grating with a period of 50 pixel:
    err = slm.showGratingBlazeHorizontal(50)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showGratingBlazeHorizontal(50)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a vertical blazed grating with a period of 50 pixel:
    err = slm.showGratingBlazeVertical(50)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showGratingBlazeVertical(50)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show an axicon phase function with an inner radius of 60 pixel:
    err = slm.showAxicon(60)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showAxicon(60)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show built-in Fresnel lens phase function with an inner radius of 120 pixel, at which the phase has shifted by 2*pi radian compared to the center phase value:
    err = slm.showPhaseFunctionLens(120)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    print("slm.showPhaseFunctionLens(120)")
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

    # Show a vortex of order one:
    err = slm.showVortex(1)
    print("slm.showVortex(1)")
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    if not slm.wait(2500) == HEDSERR_NoError:   exit(1)

# Closes everything, i.e. closes all open SLMs and SLM windows including tray icons etc., and cleans up everything:
HEDS.SDK.Close()

print("\nBuilt-in slideshow finished.")

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

