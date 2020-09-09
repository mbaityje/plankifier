#!/bin/bash
#
# ONLY LAUNCH FROM SIAM CLUSTER
#
# This script transfers the data folders from Marco's office PC to
# here (Siam20). The reason is that Marco's computer is connected to
# the disks with the data, whereas Siam20 is not.
#
#######################################################################

rsync -auvr ufficio:/home/marco/Dropbox/PROJECTS/MACHINE-LEARNING/AQUASCOPE/plankifier/data/1_zooplankton_0p5x/training /local/groups/baitygroup/plankifier/data/1_zooplankton_0p5x/
rsync -auvr ufficio:/home/marco/Dropbox/PROJECTS/MACHINE-LEARNING/AQUASCOPE/plankifier/data/1_zooplankton_0p5x/validation /local/groups/baitygroup/plankifier/data/1_zooplankton_0p5x/
