"""Generate a new Pointwise mesh recipe in the chosen empty work directory."""
import argparse
import os
from pathlib import Path
import shutil
import numpy as np
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--airfoil', choices=['NACA0018','NACA0021','cylinder','flat_plate'], default='NACA0018')
parser.add_argument('--aoa', type=float, nargs='+', required=True)
parser.add_argument('--numpts', type=int, required=True, help='point count expected by the Glyph recipe')
parser.add_argument('--pointwise', default='pointwise')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=False)
shutil.copy2(here/'IEA_Airfoil_Mesh.glf', args.output/'IEA_Airfoil_Mesh.glf')
shutil.copy2(here/f'{args.airfoil}_coords.dat', args.output/f'{args.airfoil}_coords.dat')
(args.output/'grids').mkdir()
os.chdir(args.output)
aoa_array=args.aoa
numpts=args.numpts
af_names=[args.airfoil]
for af_name in af_names:
     for i, aoa in enumerate(aoa_array):
     
          with open('IEA_Airfoil_Mesh.glf','r') as firstfile, open('tmp_grid_'+str(aoa)+'.glf','x') as secondfile: 
               
               # read content from first file 
               for line in firstfile: 
                         
                    # write content to second file 
                    secondfile.write(line)

     for i, aoa in enumerate(aoa_array):
          print("Airfoil: ", af_name, "aoa: ", aoa)
          with open('tmp_grid_'+str(aoa)+'.glf','r') as file:

               data = file.read()
               data = data.replace("set numpts 335", "set numpts "+str(numpts))
               data = data.replace("set aoa 32", "set aoa "+str(aoa))       
               data = data.replace("tmp_coords.dat", af_name+"_coords.dat")
               data = data.replace("tmp_name", af_name)
               
               if aoa < 0:
                    data = data.replace("set percent_ule 1",  "set percent_ule 50")       
                    data = data.replace("set percent_lle 50",  "set percent_lle 1")       
                    
               if np.abs(aoa) > 110: 
                    data = data.replace("$_TMP(PW_5) setName inlet", "$_TMP(PW_5) setName outlet")    
                    data = data.replace("$_TMP(PW_3) setName outlet", "$_TMP(PW_3) setName inlet")
          
          with open('tmp_grid_'+str(aoa)+'.glf', 'w') as file: 
          
               # Writing the replaced data in our 
               # text file 
               file.write(data) 

          subprocess.run([args.pointwise, "-b", "tmp_grid_"+str(aoa)+".glf"], check=True)
