# Convert output of fusibile to DTU evaluation format.
# By: Jiayu Yang
# Date: 2020-03-30

import os
from os import listdir

fusibile_out_folder="../outputs-dtu/"
dtu_eval_folder="../outputs-dtu/"

if not os.path.isdir(dtu_eval_folder):
    os.mkdir(dtu_eval_folder)

scans = ["scan1", "scan4", "scan9", "scan10", "scan11",
         "scan12", "scan13", "scan15", "scan23", "scan24",
         "scan29", "scan32", "scan33", "scan34", "scan48",
         "scan49", "scan62", "scan75", "scan77", "scan110",
         "scan114", "scan118"]

for scan in scans:
    # Move ply to dtu eval folder and rename
    scan_folder = os.path.join(fusibile_out_folder, scan, "points_mvsnet")
    consis_folders = [f for f in listdir(scan_folder) if f.startswith('consistencyCheck-')]
    
    consis_folders.sort()
    consis_folder = consis_folders[-1]
    source_ply = os.path.join(fusibile_out_folder, scan, "points_mvsnet", consis_folder, 'final3d_model.ply')
    #print("source :{}".format(source_ply))
    #source_ply = os.path.join(fusibile_out_folder,scan,'consistencyCheck/final3d_model.ply')
    scan_idx = int(scan[4:])
    target_ply = os.path.join(dtu_eval_folder,'mvsnet{:03d}_l3.ply'.format(scan_idx))

    cmd = 'mv '+source_ply+' '+target_ply

    print(cmd)
    os.system(cmd)
