#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

# ===============================================
# NACA 4-SERIES AIRFOIL GENERATOR AND 
# BOUNDARY LAYER MESH GENERATOR
# ===============================================
# Written by Travis Carrigan
#
# v1: Jan 14, 2011
# v2: Sep 21, 2011
# v3: Oct 06, 2011
#
# =============================================
# Modified for IEA 15MW airfoils
# Author: Shreyas Bidadi
# Date: 01/09/2024 

# Load Pointwise Glyph package and Tk
package require PWI_Glyph 2.4
pw::Script loadTk
pw::Application reset

# For ffa-w3-211 airfoil at Re=10M
set initds 0.00000278
set cellgr_near 1.04
set cellgr_away 1.09
set bldist 0.4
set numpts 335
set lfac 1000
set tfac 1000
set nupperfac 0.5
set nspan 121
set aoa 32 

########################################################
set fname ./tmp_coords.dat
pw::Database import $fname
    set imported 2
########################################################

# BOUNDARY LAYER MESH GENERATION PROCEDURE
# -----------------------------------------------
proc airfoilMesh {} {

# BOUNDARY LAYER INPUTS
# -----------------------------------------------
# initDs = initial cell height
# cellGr = cell growth rate
# blDist = boundary layer distance
# numPts = number of points around airfoil
# lFac = LE factor spacing factor (multiplies initDs)
# tFac = TE factor spacing factor (multiplies initDs)
# nUpperFac = factor of points on the upper surface
set initDs $::initds
set cellGrNear $::cellgr_near
set cellGrAway $::cellgr_away
set blDist $::bldist
set numPts $::numpts
set lFac $::lfac
set tFac $::tfac
set nUpperFac $::nupperfac

# CONNECTOR CREATION, DIMENSIONING, AND SPACING
# -----------------------------------------------
# Get all database entities
set dbEnts [pw::Database getAll]

# Get the curve length of all db curves
foreach db $dbEnts {
    lappend crvLength [$db getLength 1.0]
}

# Find trailing edge from minimum curve length
if {[lindex $crvLength 0] < [lindex $crvLength 1]} {
    set min 0
} else {
    set min 1
}

if {[lindex $crvLength $min] < [lindex $crvLength 2]} {
    set min $min
} else {
    set min 2
}

set dbTe [lindex $dbEnts $min]

# Get upper and lower surfaces
foreach db $dbEnts {
    if {$db != $dbTe} {
        lappend upperLower $db
    }
}

# Find y values at 50 percent length of upper and lower surfaces
set y1 [lindex [[lindex $upperLower 0] getXYZ -arc 0.5] 1]
set y2 [lindex [[lindex $upperLower 1] getXYZ -arc 0.5] 1]

# Determine upper and lower surface db entities
if {$y1 < $y2} {
    set dbLower [lindex $upperLower 0]
    set dbUpper [lindex $upperLower 1]
} else {
    set dbLower [lindex $upperLower 1]
    set dbUpper [lindex $upperLower 0]
}

# Create connectors on database entities

# Upper surface connector
set upperSurfCon [pw::Connector createOnDatabase $dbUpper]
set percent_ule 1
set _TMP(split_params) [list]
lappend _TMP(split_params) [$upperSurfCon getParameter -arc [expr {0.01 * $percent_ule}]]
set _TMP(PW_1) [$upperSurfCon split $_TMP(split_params)]
unset _TMP(PW_1)
unset _TMP(split_params)
set upperSurfCon1 [pw::GridEntity getByName con-1-split-1]
set upperSurfCon2 [pw::GridEntity getByName con-1-split-2]
$upperSurfCon1 setName con-1
$upperSurfCon2 setName con-2

# Lower surface connector
set lowerSurfCon [pw::Connector createOnDatabase $dbLower]
set percent_lle 50
set _TMP(split_params) [list]
lappend _TMP(split_params) [$lowerSurfCon getParameter -arc [expr {0.01 * $percent_lle}]]
set _TMP(PW_1) [$lowerSurfCon split $_TMP(split_params)]
unset _TMP(PW_1)
unset _TMP(split_params)
set lowerSurfCon1 [pw::GridEntity getByName con-3-split-1]
set lowerSurfCon2 [pw::GridEntity getByName con-3-split-2]
$lowerSurfCon1 setName con-3
$lowerSurfCon2 setName con-4

# Trailing surface connector
set trailSurfCon [pw::Connector createOnDatabase $dbTe]
set cons "$upperSurfCon1 $upperSurfCon2 $lowerSurfCon1 $lowerSurfCon2 $trailSurfCon"

# Calculate main airfoil connector dimensions
foreach con $cons {lappend conLen [$con getLength -arc 1]}
set upperSurfConLen1 [lindex $conLen 0]
set upperSurfConLen2 [lindex $conLen 1]
set lowerSurfConLen1 [lindex $conLen 2]
set lowerSurfConLen2 [lindex $conLen 3]
set trailSurfConLen [lindex $conLen 4]
set conDimUpper1 [expr int(10)]
set conDimUpper2 [expr int($numPts*$nUpperFac)]
set conDimLower1 [expr int(10)]
set conDimLower2 [expr int($numPts*(1-$nUpperFac))]

# Dimension upper and lower airfoil surface connectors
$upperSurfCon1 setDimension $conDimUpper1
$upperSurfCon2 setDimension $conDimUpper2
$lowerSurfCon1 setDimension $conDimLower1
$lowerSurfCon2 setDimension $conDimLower2

# Dimension trailing edge airfoil connector
set teDim [expr int($trailSurfConLen/($tFac*$initDs))+2]
$trailSurfCon setDimension $teDim
#puts $trailSurfConLen
#puts $tFac
#puts $initDs
#puts $teDim

# Set leading and trailing edge connector spacings
set lDs [expr $lFac*$initDs]
set tDs [expr $tFac*$initDs]

# Dimension the LE uniform section
set leDim [expr int($upperSurfConLen1/($lDs))+2]
$upperSurfCon1 setDimension $leDim
set leDim [expr int($lowerSurfConLen1/($lDs))+2]
$lowerSurfCon1 setDimension $leDim

# Set the begin and end spacings
set upperSurfConDis2 [$upperSurfCon2 getDistribution 1]
set lowerSurfConDis2 [$lowerSurfCon2 getDistribution 1]
set trailSurfConDis [$trailSurfCon getDistribution 1]

$upperSurfConDis2 setBeginSpacing $lDs
$upperSurfConDis2 setEndSpacing $tDs
$lowerSurfConDis2 setBeginSpacing $lDs
$lowerSurfConDis2 setEndSpacing $tDs

# Create edges for structured boundary layer extrusion
set upperSurfCon [pw::Connector join -reject _TMP(ignored) -keepDistribution [list $upperSurfCon2 $upperSurfCon1]]
set lowerSurfCon [pw::Connector join -reject _TMP(ignored) -keepDistribution [list $lowerSurfCon2 $lowerSurfCon1]]
set cons "$upperSurfCon $lowerSurfCon $trailSurfCon "
set afEdge [pw::Edge createFromConnectors -single $cons]
set afDom [pw::DomainStructured create]
$afDom addEdge $afEdge

# Extrude boundary layer using normal hyperbolic extrusion method
set afExtrude [pw::Application begin ExtrusionSolver $afDom]
	$afDom setExtrusionSolverAttribute NormalInitialStepSize $initDs
	$afDom setExtrusionSolverAttribute SpacingGrowthFactor $cellGrNear
	$afDom setExtrusionSolverAttribute NormalMarchingVector {0 0 -1}
	$afDom setExtrusionSolverAttribute NormalKinseyBarthSmoothing 3
	$afDom setExtrusionSolverAttribute NormalVolumeSmoothing 0.3
	$afDom setExtrusionSolverAttribute StopAtHeight $blDist
	$afExtrude run 1000
	$afDom setExtrusionSolverAttribute NormalKinseyBarthSmoothing 0
	$afDom setExtrusionSolverAttribute NormalVolumeSmoothing 1.0
	$afDom setExtrusionSolverAttribute SpacingGrowthFactor $cellGrAway
	$afDom setExtrusionSolverAttribute StopAtHeight 60.0
	$afExtrude run 1000
$afExtrude end

# Rename some connectors
$upperSurfCon setName con-1
$lowerSurfCon setName con-2
$trailSurfCon setName con-3
set extrudeCon [pw::GridEntity getByName con-6]
$extrudeCon setName con-4
set outerCon [pw::GridEntity getByName con-7]
$outerCon setName con-5

# Reset view
pw::Display resetView

}

proc extrudeMesh {} {

set nSpan $::nspan
set aoa $::aoa

set _TMP(mode_1) [pw::Application begin Create]
  set _TMP(PW_1) [pw::SegmentSpline create]
  set _DM(1) [pw::GridEntity getByName dom-1]
  set _CN(1) [pw::GridEntity getByName con-3]
  set _DB(1) [pw::DatabaseEntity getByName curve-1]
  set _DB(2) [pw::DatabaseEntity getByName curve-3]
  set _CN(2) [pw::GridEntity getByName con-1]
  set _DB(3) [pw::DatabaseEntity getByName curve-2]
  set _CN(3) [pw::GridEntity getByName con-2]
  set _CN(4) [pw::GridEntity getByName con-4]
  $_TMP(PW_1) addPoint [$_CN(2) getPosition -arc 0]
  $_TMP(PW_1) addPoint {0 0 4}
  set _CN(5) [pw::Connector create]
  $_CN(5) addSegment $_TMP(PW_1)
  unset _TMP(PW_1)
  $_CN(5) calculateDimension
$_TMP(mode_1) end
$_CN(5) setName con-6
unset _TMP(mode_1)
pw::Application markUndoLevel {Create 2 Point Connector}

set _TMP(mode_1) [pw::Application begin Create]
  set _TMP(PW_1) [pw::SegmentSpline create]
  $_TMP(PW_1) delete
  unset _TMP(PW_1)
$_TMP(mode_1) abort
unset _TMP(mode_1)
$_CN(5) setDimension $nSpan
pw::CutPlane refresh
pw::Application markUndoLevel Dimension

set _TMP(split_params) [list]
set splitIndex [lindex [$_CN(5) closestCoordinate {0 0 2}] 0]
set splitIndexP1 [expr $splitIndex+1]
lappend _TMP(split_params) $splitIndex
lappend _TMP(split_params) $splitIndexP1
set _TMP(PW_1) [$_CN(5) split -I $_TMP(split_params)]
unset _TMP(PW_1)
unset _TMP(split_params)
pw::Application markUndoLevel Split

set nSteps [expr $nSpan-1]

set _TMP(mode_1) [pw::Application begin Create]
  set _TMP(PW_1) [pw::FaceStructured createFromDomains [list $_DM(1)]]
  set _TMP(face_1) [lindex $_TMP(PW_1) 0]
  unset _TMP(PW_1)
  set _BL(1) [pw::BlockStructured create]
  $_BL(1) addFace $_TMP(face_1)
$_TMP(mode_1) end
unset _TMP(mode_1)
set _TMP(mode_1) [pw::Application begin ExtrusionSolver [list $_BL(1)]]
  $_TMP(mode_1) setKeepFailingStep true
  $_BL(1) setExtrusionSolverAttribute Mode Path
  set _CN(6) [pw::GridEntity getByName con-6-split-1]
  set _CN(7) [pw::GridEntity getByName con-6-split-2]
  set _CN(8) [pw::GridEntity getByName con-6-split-3]
  $_BL(1) setExtrusionSolverAttribute PathConnectors [list $_CN(6) $_CN(7) $_CN(8)]
  $_BL(1) setExtrusionSolverAttribute PathUseTangent 1
  $_TMP(mode_1) run $nSteps
$_TMP(mode_1) end
unset _TMP(mode_1)
unset _TMP(face_1)
pw::Application markUndoLevel {Extrude, Path}

pw::Display resetView -Z
pw::Display resetView -X
set _DM(2) [pw::GridEntity getByName dom-3]
set _DM(3) [pw::GridEntity getByName dom-2]
set _DM(4) [pw::GridEntity getByName dom-8]
set _TMP(split_params) [list]
lappend _TMP(split_params) [lindex [list 50 $splitIndex $_DM(2)] 1]
lappend _TMP(split_params) [lindex [list 50 $splitIndexP1 $_DM(2)] 1]
set _TMP(PW_1) [$_DM(2) split -J $_TMP(split_params)]
unset _TMP(PW_1)
unset _TMP(split_params)
pw::Application markUndoLevel Split

set _DM(5) [pw::GridEntity getByName dom-3-split-3]
set _DM(6) [pw::GridEntity getByName dom-6]
set _DM(7) [pw::GridEntity getByName dom-3-split-2]
set _DM(8) [pw::GridEntity getByName dom-3-split-1]
set _CN(9) [pw::GridEntity getByName con-15]
set _CN(10) [pw::GridEntity getByName con-16]
set _TMP(split_params) [list]
lappend _TMP(split_params) [lindex [list 77 $splitIndex $_DM(3)] 1]
lappend _TMP(split_params) [lindex [list 77 $splitIndexP1 $_DM(3)] 1]
set _TMP(PW_1) [$_DM(3) split -J $_TMP(split_params)]
unset _TMP(PW_1)
unset _TMP(split_params)
pw::Application markUndoLevel Split

set _DM(9) [pw::GridEntity getByName dom-4]
set _DM(10) [pw::GridEntity getByName dom-2-split-3]
set _DM(11) [pw::GridEntity getByName dom-5]
set _DM(12) [pw::GridEntity getByName dom-2-split-2]
set _CN(11) [pw::GridEntity getByName con-18]
set _TMP(split_params) [list]
lappend _TMP(split_params) [lindex [$_DM(9) closestCoordinate [pw::Grid getPoint [list 63 $splitIndex $_DM(11)]]] 1]
lappend _TMP(split_params) [lindex [$_DM(9) closestCoordinate [pw::Grid getPoint [list 60 $splitIndexP1 $_DM(11)]]] 1]
set _TMP(PW_1) [$_DM(9) split -J $_TMP(split_params)]
unset _TMP(PW_1)
unset _TMP(split_params)
pw::Application markUndoLevel Split


set rotAngle [expr -$aoa]

set _CN(12) [pw::GridEntity getByName con-10-split-1]
set _CN(13) [pw::GridEntity getByName con-10-split-2-split-1]
set _CN(14) [pw::GridEntity getByName con-10-split-2-split-2]
set _CN(15) [pw::GridEntity getByName con-5]
set _CN(16) [pw::GridEntity getByName con-8-split-1]
set _CN(17) [pw::GridEntity getByName con-8-split-2-split-1]
set _CN(18) [pw::GridEntity getByName con-8-split-2-split-2]
set _CN(19) [pw::GridEntity getByName con-17]
set _CN(20) [pw::GridEntity getByName con-20]
set _CN(21) [pw::GridEntity getByName con-19]
set _CN(22) [pw::GridEntity getByName con-6]
set _CN(23) [pw::GridEntity getByName con-9]
set _CN(24) [pw::GridEntity getByName con-12]
set _CN(25) [pw::GridEntity getByName con-11]
set _CN(26) [pw::GridEntity getByName con-13]
set _CN(27) [pw::GridEntity getByName con-14]
set _DM(13) [pw::GridEntity getByName dom-4-split-1]
set _DM(14) [pw::GridEntity getByName dom-4-split-3]
set _DM(15) [pw::GridEntity getByName dom-4-split-2]
set _DM(16) [pw::GridEntity getByName dom-2-split-1]
set _TMP(mode_1) [pw::Application begin Modify [list $_DB(1) $_DB(3) $_DB(2) $_CN(1) $_CN(3) $_CN(12) $_CN(4) $_CN(13) $_CN(2) $_CN(10) $_CN(14) $_CN(15) $_CN(9) $_CN(16) $_CN(17) $_CN(18) $_CN(11) $_CN(19) $_CN(20) $_CN(21) $_CN(22) $_CN(23) $_CN(24) $_CN(25) $_CN(26) $_CN(27) $_CN(8) $_CN(6) $_CN(7) $_DM(1) $_DM(8) $_DM(5) $_DM(7) $_BL(1) $_DM(13) $_DM(14) $_DM(15) $_DM(10) $_DM(6) $_DM(11) $_DM(4) $_DM(12) $_DM(16)]]
  pw::Display resetView -Z
  pw::Entity transform [pwu::Transform rotation -anchor {0 0 0} {0 0 1} $rotAngle] [$_TMP(mode_1) getEntities]
$_TMP(mode_1) end
unset _TMP(mode_1)
pw::Application markUndoLevel Rotate

set _TMP(split_params) [list]
lappend _TMP(split_params) [lindex [$_DM(6) closestCoordinate {5 50 0}] 0]
lappend _TMP(split_params) [lindex [$_DM(6) closestCoordinate {5 -50 0}] 0]
set _TMP(PW_1) [$_DM(6) split -I $_TMP(split_params)]
unset _TMP(PW_1)
unset _TMP(split_params)
pw::Application markUndoLevel Split

set _TMP(PW_1) [pw::VolumeCondition create]
pw::Application markUndoLevel {Create VC}

$_TMP(PW_1) setPhysicalType Fluid
pw::Application markUndoLevel {Change VC Type}

unset _TMP(PW_1)
pw::Application markUndoLevel {Set Solver Attributes}

pw::Application setCAESolver {EXODUS II} 3
pw::Application markUndoLevel {Select Solver}

set _TMP(PW_1) [pw::VolumeCondition getByName vc-2]
$_TMP(PW_1) setName fluid
pw::Application markUndoLevel {Name VC}

$_TMP(PW_1) apply [list $_BL(1)]
pw::Application markUndoLevel {Set VC}

unset _TMP(PW_1)
set _DM(17) [pw::GridEntity getByName dom-6-split-1]
set _DM(18) [pw::GridEntity getByName dom-6-split-2]
set _DM(19) [pw::GridEntity getByName dom-6-split-3]
set _TMP(PW_1) [pw::BoundaryCondition getByName Unspecified]
set _TMP(PW_2) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

set _TMP(PW_3) [pw::BoundaryCondition getByName bc-2]
unset _TMP(PW_2)
set _TMP(PW_4) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

set _TMP(PW_5) [pw::BoundaryCondition getByName bc-3]
unset _TMP(PW_4)
set _TMP(PW_6) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

set _TMP(PW_7) [pw::BoundaryCondition getByName bc-4]
unset _TMP(PW_6)
set _TMP(PW_8) [pw::BoundaryCondition create]
pw::Application markUndoLevel {Create BC}

set _TMP(PW_9) [pw::BoundaryCondition getByName bc-5]
unset _TMP(PW_8)
$_TMP(PW_3) setName outlet
$_TMP(PW_3) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Name BC}

$_TMP(PW_5) setName inlet
$_TMP(PW_5) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Name BC}

$_TMP(PW_7) setName wing
$_TMP(PW_7) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Name BC}

$_TMP(PW_9) setName wing-pp
$_TMP(PW_9) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Name BC}

set _TMP(PW_10) [pw::BoundaryCondition create]
$_TMP(PW_10) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Create BC}

set _TMP(PW_11) [pw::BoundaryCondition getByName bc-6]
unset _TMP(PW_10)
$_TMP(PW_11) setName front
$_TMP(PW_11) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Name BC}

set _TMP(PW_12) [pw::BoundaryCondition create]
$_TMP(PW_12) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Create BC}

set _TMP(PW_13) [pw::BoundaryCondition getByName bc-7]
unset _TMP(PW_12)
$_TMP(PW_13) setName back
$_TMP(PW_13) setPhysicalType -usage CAE {Side Set}
pw::Application markUndoLevel {Name BC}

$_TMP(PW_11) apply [list [list $_BL(1) $_DM(4)]]
pw::Application markUndoLevel {Set BC}

$_TMP(PW_13) apply [list [list $_BL(1) $_DM(1)]]
pw::Application markUndoLevel {Set BC}

$_TMP(PW_5) apply [list [list $_BL(1) $_DM(18)]]
pw::Application markUndoLevel {Set BC}

$_TMP(PW_3) apply [list [list $_BL(1) $_DM(17)] [list $_BL(1) $_DM(19)]]
pw::Application markUndoLevel {Set BC}

$_TMP(PW_7) apply [list [list $_BL(1) $_DM(10)] [list $_BL(1) $_DM(13)] [list $_BL(1) $_DM(8)] [list $_BL(1) $_DM(16)] [list $_BL(1) $_DM(14)] [list $_BL(1) $_DM(5)]]
pw::Application markUndoLevel {Set BC}

$_TMP(PW_9) apply [list [list $_BL(1) $_DM(15)] [list $_BL(1) $_DM(7)] [list $_BL(1) $_DM(12)]]
pw::Application markUndoLevel {Set BC}

unset _TMP(PW_1)
unset _TMP(PW_3)
unset _TMP(PW_5)
unset _TMP(PW_7)
unset _TMP(PW_9)
unset _TMP(PW_11)
unset _TMP(PW_13)
pw::Application save ./grids/tmp_name$aoa.pw
set _TMP(mode_1) [pw::Application begin CaeExport [pw::Entity sort [list $_BL(1)]]]
  $_TMP(mode_1) initialize -strict -type CAE ./grids/tmp_name$aoa.exo
  $_TMP(mode_1) verify
  $_TMP(mode_1) write
$_TMP(mode_1) end
unset _TMP(mode_1)

}

# PROCEDURE TO DELETE ANY EXISTING GRID ENTITIES
# -----------------------------------------------
proc cleanGrid {} {

    set grids [pw::Grid getAll -type pw::Connector]

    if {[llength $grids]>0} {
        foreach grid $grids {$grid delete -force}
    }

}

# PROCEDURE TO DELETE ANY EXISTING GEOMETRY
# -----------------------------------------------
proc cleanGeom {} {

    cleanGrid    

    set dbs [pw::Database getAll]
    
    if {[llength $dbs]>0} {
        foreach db $dbs {$db delete -force}
    }

}

#####################################
cleanGrid
puts "cleanGrid"

airfoilMesh
puts "airfoilMesh"

extrudeMesh
puts "extrudeMesh"
#####################################

pw::Application exit

# END SCRIPT

#############################################################################
#
# This file is licensed under the Cadence Public License Version 1.0 (the
# "License"), a copy of which is found in the included file named "LICENSE",
# and is distributed "AS IS." TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE
# LAW, CADENCE DISCLAIMS ALL WARRANTIES AND IN NO EVENT SHALL BE LIABLE TO
# ANY PARTY FOR ANY DAMAGES ARISING OUT OF OR RELATING TO USE OF THIS FILE.
# Please see the License for the full text of applicable terms.
#
#############################################################################
