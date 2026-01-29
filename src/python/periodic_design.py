# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
import numpy as np
import argparse
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def main(**kwargs):
    vtufile = kwargs["vtufile"]
    L = kwargs["L"]
    W = kwargs["W"]
    Nrow = kwargs["Nrow"]
    Ncol = kwargs["Ncol"]
    savepath = kwargs["savepath"]
    orientation = kwargs["orientation"]
        
    # create a new 'XML Unstructured Grid Reader'
    designvtu = XMLUnstructuredGridReader(registrationName=vtufile, FileName=[vtufile])
    designvtu.CellArrayStatus = ['cell']
    designvtu.PointArrayStatus = ['p']

    # Properties modified on designvtu
    designvtu.CellArrayStatus = []
    designvtu.TimeArray = 'None'

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    designvtuDisplay = Show(designvtu, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    designvtuDisplay.Representation = 'Surface'
    designvtuDisplay.ColorArrayName = [None, '']
    designvtuDisplay.SelectTCoordArray = 'None'
    designvtuDisplay.SelectNormalArray = 'None'
    designvtuDisplay.SelectTangentArray = 'None'
    designvtuDisplay.OSPRayScaleArray = 'p'
    designvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    designvtuDisplay.SelectOrientationVectors = 'None'
    designvtuDisplay.ScaleFactor = 20.0
    designvtuDisplay.SelectScaleArray = 'None'
    designvtuDisplay.GlyphType = 'Arrow'
    designvtuDisplay.GlyphTableIndexArray = 'None'
    designvtuDisplay.GaussianRadius = 1.0
    designvtuDisplay.SetScaleArray = ['POINTS', 'p']
    designvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    designvtuDisplay.OpacityArray = ['POINTS', 'p']
    designvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    designvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
    designvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
    designvtuDisplay.ScalarOpacityUnitDistance = 10.69043801605173
    designvtuDisplay.OpacityArrayName = ['POINTS', 'p']
    designvtuDisplay.SelectInputVectors = [None, '']
    designvtuDisplay.WriteLog = ''

    # reset view to fit data
    renderView1.ResetCamera(False)

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()

    # change representation type
    designvtuDisplay.SetRepresentationType('Surface')

    # set scalar coloring
    ColorBy(designvtuDisplay, ('POINTS', 'p'))

    # rescale color and/or opacity maps used to include current data range
    designvtuDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    designvtuDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'p'
    pLUT = GetColorTransferFunction('p')
    pLUT.ApplyPreset('X Ray', True)

    # get opacity transfer function/opacity map for 'p'
    pPWF = GetOpacityTransferFunction('p')

    # get 2D transfer function for 'p'
    pTF2D = GetTransferFunction2D('p')

    if Nrow > 0 and Ncol > 0:
        # create a new 'Reflect'
        reflect1 = Reflect(registrationName='Reflect1', Input=designvtu)

        # show data in view
        reflect1Display = Show(reflect1, renderView1, 'UnstructuredGridRepresentation')

        # trace defaults for the display properties.
        reflect1Display.Representation = 'Surface'
        reflect1Display.ColorArrayName = ['POINTS', 'p']
        reflect1Display.LookupTable = pLUT
        reflect1Display.SelectTCoordArray = 'None'
        reflect1Display.SelectNormalArray = 'None'
        reflect1Display.SelectTangentArray = 'None'
        reflect1Display.OSPRayScaleArray = 'p'
        reflect1Display.OSPRayScaleFunction = 'PiecewiseFunction'
        reflect1Display.SelectOrientationVectors = 'None'
        reflect1Display.ScaleFactor = 40.0
        reflect1Display.SelectScaleArray = 'None'
        reflect1Display.GlyphType = 'Arrow'
        reflect1Display.GlyphTableIndexArray = 'None'
        reflect1Display.GaussianRadius = 2.0
        reflect1Display.SetScaleArray = ['POINTS', 'p']
        reflect1Display.ScaleTransferFunction = 'PiecewiseFunction'
        reflect1Display.OpacityArray = ['POINTS', 'p']
        reflect1Display.OpacityTransferFunction = 'PiecewiseFunction'
        reflect1Display.DataAxesGrid = 'GridAxesRepresentation'
        reflect1Display.PolarAxes = 'PolarAxesRepresentation'
        reflect1Display.ScalarOpacityFunction = pPWF
        reflect1Display.ScalarOpacityUnitDistance = 11.999610800238877
        reflect1Display.OpacityArrayName = ['POINTS', 'p']
        reflect1Display.SelectInputVectors = [None, '']
        reflect1Display.WriteLog = ''

        # hide data in view
        Hide(designvtu, renderView1)

        # show color bar/color legend
        reflect1Display.SetScalarBarVisibility(renderView1, True)

        # update the view to ensure updated data information
        renderView1.Update()

        # create a new 'Reflect'
        reflect2 = Reflect(registrationName='Reflect2', Input=reflect1)

        # Properties modified on reflect2
        reflect2.Plane = 'Y Min'

        # show data in view
        reflect2Display = Show(reflect2, renderView1, 'UnstructuredGridRepresentation')

        # trace defaults for the display properties.
        reflect2Display.Representation = 'Surface'
        reflect2Display.ColorArrayName = ['POINTS', 'p']
        reflect2Display.LookupTable = pLUT
        reflect2Display.SelectTCoordArray = 'None'
        reflect2Display.SelectNormalArray = 'None'
        reflect2Display.SelectTangentArray = 'None'
        reflect2Display.OSPRayScaleArray = 'p'
        reflect2Display.OSPRayScaleFunction = 'PiecewiseFunction'
        reflect2Display.SelectOrientationVectors = 'None'
        reflect2Display.ScaleFactor = 40.0
        reflect2Display.SelectScaleArray = 'None'
        reflect2Display.GlyphType = 'Arrow'
        reflect2Display.GlyphTableIndexArray = 'None'
        reflect2Display.GaussianRadius = 2.0
        reflect2Display.SetScaleArray = ['POINTS', 'p']
        reflect2Display.ScaleTransferFunction = 'PiecewiseFunction'
        reflect2Display.OpacityArray = ['POINTS', 'p']
        reflect2Display.OpacityTransferFunction = 'PiecewiseFunction'
        reflect2Display.DataAxesGrid = 'GridAxesRepresentation'
        reflect2Display.PolarAxes = 'PolarAxesRepresentation'
        reflect2Display.ScalarOpacityFunction = pPWF
        reflect2Display.ScalarOpacityUnitDistance = 11.664589499322732
        reflect2Display.OpacityArrayName = ['POINTS', 'p']
        reflect2Display.SelectInputVectors = [None, '']
        reflect2Display.WriteLog = ''

        # hide data in view
        Hide(reflect1, renderView1)

        # show color bar/color legend
        reflect2Display.SetScalarBarVisibility(renderView1, True)

        # update the view to ensure updated data information
        renderView1.Update()

        # Build references
        transforms = [[None]*Ncol for _ in range(Nrow)]
        translate_x = np.linspace(0, L*(Ncol-1), Ncol)
        translate_y = np.linspace(0, W*(Nrow-1), Nrow)
        for i, ty in enumerate(translate_y):
            for j, tx in enumerate(translate_x):

                # create a new 'Transform'
                transform = Transform(registrationName='Transform_'+str(i)+"_"+str(j), Input=reflect2)
                transforms[i][j] = transform
                transform.Transform = 'Transform'

                # Properties modified on transform.Transform
                transform.Transform.Translate = [tx, ty, 0.0]

                # show data in view
                transformDisplay = Show(transform, renderView1, 'UnstructuredGridRepresentation')

                # trace defaults for the display properties.
                transformDisplay.Representation = 'Surface'
                transformDisplay.ColorArrayName = ['POINTS', 'p']
                transformDisplay.LookupTable = pLUT
                transformDisplay.SelectTCoordArray = 'None'
                transformDisplay.SelectNormalArray = 'None'
                transformDisplay.SelectTangentArray = 'None'
                transformDisplay.OSPRayScaleArray = 'p'
                transformDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
                transformDisplay.SelectOrientationVectors = 'None'
                transformDisplay.ScaleFactor = 40.0
                transformDisplay.SelectScaleArray = 'None'
                transformDisplay.GlyphType = 'Arrow'
                transformDisplay.GlyphTableIndexArray = 'None'
                transformDisplay.GaussianRadius = 2.0
                transformDisplay.SetScaleArray = ['POINTS', 'p']
                transformDisplay.ScaleTransferFunction = 'PiecewiseFunction'
                transformDisplay.OpacityArray = ['POINTS', 'p']
                transformDisplay.OpacityTransferFunction = 'PiecewiseFunction'
                transformDisplay.DataAxesGrid = 'GridAxesRepresentation'
                transformDisplay.PolarAxes = 'PolarAxesRepresentation'
                transformDisplay.ScalarOpacityFunction = pPWF
                transformDisplay.ScalarOpacityUnitDistance = 11.664589499322732
                transformDisplay.OpacityArrayName = ['POINTS', 'p']
                transformDisplay.SelectInputVectors = [None, '']
                transformDisplay.WriteLog = ''

                # show color bar/color legend
                transformDisplay.SetScalarBarVisibility(renderView1, True)

                # update the view to ensure updated data information
                renderView1.Update()

                # set active source
                SetActiveSource(reflect2)

                # toggle interactive widget visibility (only when running from the GUI)
                HideInteractiveWidgets(proxy=transform.Transform)
        
        # Hide the main view
        Hide(reflect2, renderView1)
        reflect2Display.SetScalarBarVisibility(renderView1, False)
    else:
        reflect2Display = Show(designvtu, renderView1, 'UnstructuredGridRepresentation')
        reflect2Display.SetScalarBarVisibility(renderView1, False)
    
    if orientation == "x":
        renderView1.ResetActiveCameraToPositiveX()
    elif orientation == "mx":
        renderView1.ResetActiveCameraToNegativeX()
    elif orientation == "y":
        renderView1.ResetActiveCameraToPositiveY()
    elif orientation == "my":
        renderView1.ResetActiveCameraToNegativeY()
    elif orientation == "z":
        renderView1.ResetActiveCameraToPositiveZ()
    else:
        renderView1.ResetActiveCameraToNegativeZ()
    renderView1.ResetCamera(False)
        
    SaveScreenshot(savepath, renderView1, ImageResolution= (4200, 3876))


parser = argparse.ArgumentParser()
parser.add_argument("--vtufile", type=str, default="/Users/ianhammond/GitHub/Emitter3DTopOpt/design.vtu")
parser.add_argument("--L", type=float, default=400)
parser.add_argument("--W", type=float, default=400)
parser.add_argument("--Nrow", type=int, default=1)
parser.add_argument("--Ncol", type=int, default=None)
parser.add_argument("--savepath", type=str, default="/Users/ianhammond/GitHub/Emitter3DTopOpt/design.png")
parser.add_argument("--orientation", type=str, default="mz")
args = parser.parse_args()
if args.Ncol is None:
    args.Ncol = args.Nrow
main(**vars(args))