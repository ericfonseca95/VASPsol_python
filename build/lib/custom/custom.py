#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:47:07 2019

@author: funsega
"""
import numpy as np
import ase.io as io
#from dscribe.descriptors import SOAP
from ase.io import dmol 
from pymatgen.core import Structure
from pymatgen.io.xyz import XYZ
import os
import shutil as sh
from pymatgen.io.vasp.outputs import Vasprun
from dask.distributed import Client
import time
from dask_jobqueue import SLURMCluster
from pymatgen.core.periodic_table import Element
import subprocess as sp
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Vasprun, Chgcar, Locpot
import matplotlib.pyplot as plt

def Replace_runjob(tag,value):
    f1 = open('runjob',"r")
    file = f1.readlines()
    f2 = open('runjob',"w")
    n = 0
    tagline = -1
    for k in file:
        check = k.find(tag)
        if check !=-1:
            tagline = n
        n = n+1
    if tagline != -1:  
        file[tagline] = (tag + str(value) + '\n')
        for i in np.arange(len(file)):
            f2.write(file[i])
    else:
        print('Tag not Found ... adding to file now...')
        file.append(tag + str(value) + '\n')
        for i in np.arange(len(file)):
            f2.write(file[i])
    f2.close()
    f1.close()    


def Replace_Incar(tag,value):
    f1 = open('INCAR',"r")
    file = f1.readlines()
    n = 0
    f2 = open('INCAR',"w")
    tagline = -1
    check = False
    for k in file:
        check = k.find(tag)
        if check !=-1:
            tagline = n
        n = n+1
    if tagline != -1:
        if value == 'remove':
            print('deleting INCAR tag ...')
            file[tagline] = ''
            check = True
        elif value == 'check':
            print(tag,'is present')
            check = True
        else:
            file[tagline] = (tag + ' = ' + str(value) + '\n')
            check = True
        for i in np.arange(len(file)):
            f2.write(file[i])
    else:
        if value == 'remove':
            for i in np.arange(len(file)):
                f2.write(file[i])
        elif value == 'check':
            print(tag,'is not present')
            check = False
            for i in np.arange(len(file)):
                f2.write(file[i])
        else:
            print(tag+' Tag not Found ... adding to file now...')
            file.append(tag +' = '+ str(value) + '\n')
            for i in np.arange(len(file)):
                f2.write(file[i])
    f2.close()
    f1.close()
    return check
######################################################################

def aug_occupancy(filename):
	fp1 = open(filename,'r')
	lines = fp1.readlines()
	spin = 0
	matrix = []
	matrix1 = []
	matrix2 = []
	for i in np.arange(len(lines)):
		if lines[i].find('total augmentation') >-1 and spin == 0:
			spin = 1
			matrix1 = lines[i+1:i+18]
			
			for j in np.arange(i+1,i+18):
				matrix1.append(lines[j])
			
		if lines[i].find('total augmentation') >-1 and spin ==1:
			spin = 2
			
			for j in np.arange(i+1,i+18):
				matrix2.append(lines[j])
			
	
	matrix1_extract = []
	matrix2_extract = []
	for i in np.arange(len(matrix1)):
		matrix1_extract.append(list([float(s) for s in matrix1[i].split()]))
	for i in np.arange(len(matrix2)):
		matrix2_extract.append(list([float(s) for s in matrix2[i].split()]))
	matrix = [np.array(matrix1_extract),np.array(matrix2_extract)]
	fp1.close	
	return matrix

############################################################################
def occupancy(filename):
	fp1 = open(filename,'r')
	lines = fp1.readlines()
	spin = 0
	matrix = []
	matrix1 = []
	matrix2 = []
	for i in np.arange(0,len(lines))[::-1]:
		if spin == 2:
			break
		if lines[i].find('spin component  1') >-1:
			matrix1 = lines[i+2:i+7]
			spin = spin+1
			
		if lines[i].find('spin component  2') >-1:
			spin = spin+1
			matrix2 = lines[i+2:i+7]
			
	matrix1_extract = []
	matrix2_extract = []
	for i in np.arange(len(matrix1)):
		matrix1_extract.append(list([float(s) for s in matrix1[i].split()]))
	for i in np.arange(len(matrix2)):
		matrix2_extract.append(list([float(s) for s in matrix2[i].split()]))
	matrix = [np.array(matrix1_extract),np.array(matrix2_extract)]
	fp1.close	
	return matrix
###############################################################################

def HOMO_LUMO(filename):
	# imput the OUTCAR filename as a string
	fp1 = open(filename,'r')
	lines = fp1.readlines()
	dE = []
	spin = 0
	for i in np.arange(len(lines))[::-1]:
		if lines[i].find('spin component') >-1 and lines[i][16].isdigit() == True:
			j = 0
			check = 0
			spin = spin+1
			while check ==0:
				occ = float(lines[i+j+4].split()[2])
				if occ == 0:
					dE.append(float(lines[i+j+4].split()[1]) - float(lines[i+j+4-1].split()[1]))
					check = 1
				j = j+1
				if j > 200:
					break
		if spin == 2:
			break
	
	return dE
################################################################################

def o(filename):
	fp1 = open(filename,'r')
	lines = fp1.readlines()
	keep_interact = []
	keep_bare = []	
	for i in np.arange(len(lines))[::-1]:
		if lines[i].find('o =') >-1:
			for j in lines[i-15:i]:
				if j.find('o = ') >-1:
					keep_interact.append(j)
			break
	for i in np.arange(len(lines)):
		if lines[i].find('o =')>-1:
			for j in lines[i:i+15]:
				if j.find('o =') >-1:
					keep_bare.append(j)
			break
	
	o_bare = sum([float(i[7:7+6]) for i in keep_bare])
	o_interact = sum([float(i[7:7+6]) for i in keep_interact])
	o = [o_bare,o_interact]
	return o
################################################################################

def Nd(filename):
	fp1 = open(filename,'r')
	lines = fp1.readlines()
	check = 0 
	for i in np.arange(len(lines))[::-1]:
		if lines[i].find('magnetization (x)') >-1:
			j = i-1
			while check ==0:
				if lines[j].find('ion') >-1:
					check = 1
					break
				if j-i > 100:
					break
				if check ==0:
					j = j-1
			if check ==1:
				j = j+2
				check2 = 0
				k = j
				Ntemp = []
				while check2 == 0:
					check_line = lines[k].split()
					check_ion = check_line[0]
					if check_ion.isdigit() == True:
						Ntemp.append(check_line)
					else:
						check2=1
						N = Ntemp
					k = k+1
				#N_line = lines[j+2]	
				#Ntemp = N_line.split()
				break
			else:
				N_line = [0,0,0,0]
				Ntemp = N_line
				print('WARNING: TOTAL CHARGE NOT FOUND. MAKE SURE LORBIT = 11 AND LDAUPRINT = 2')
	#N = [float(i) for i in Ntemp[1::]]
	if check == 0:
		N = [0,0,0,0]
	return N



########################################################################################3

def get_U(filename):
        #provide an INCAR
        fp1 = open(filename,'r')
        lines = fp1.readlines()
        for i in lines:
                if i.find('LDAUU') >-1:
                        LDAU_LINE = i
        U = []
        for i in LDAU_LINE.split():
                try:
                        if i[0].isdigit():
                                U.append(float(i))
                except:
                        print('NO U VALUE FOUND')
        return U

###########################################################################################

def vacuum_padding(filename,amount):

	fp2 = open('POSCAR_VAC','w')
	lines = open(filename,'r').readlines()
	s = Structure.from_file(filename)
	sites = s.sites
	cart = s.cart_coords
	minA = min([d[0] for d in cart])
	maxA = max([d[0] for d in cart])
	A_dist = maxA-minA
	minB = min([d[1] for d in cart])
	maxB = max([d[1] for d in cart])
	B_dist = maxB-minB
	minC = min([d[2] for d in cart])
	maxC = max([d[2] for d in cart])
	C_dist = maxC-minC
	center_point = np.array([A_dist/2+minA,B_dist/2+minB,C_dist/2+minC])
	newC = []
	lat = s.lattice.as_dict()['matrix']
	lat[0][0] = amount+maxA-minA
	lat[1][1] = amount+maxB-minB
	lat[2][2] = amount+maxC-minC
	a = lat[0][0]
	b = lat[1][1]
	c = lat[2][2]
	trans_vector = [0.5*a,0.5*b,0.5*c] - center_point

	#print(cart[0][2])
	for j in np.arange(len(cart)):
		da = trans_vector[0]
		db = trans_vector[1]
		dc = trans_vector[2]
		x0 = cart[j][0]+da
		x1 = cart[j][1]+db
		x2 = cart[j][2]+dc
		#print(da,' ',db,' ',dc)
		#print(x0,' ',x1,' ',x2)
		newC.append([x0/a,x1/b,x2/c])

	# writes new POSCAR file as POSCAR_PAD with vacuum padding
	fp2.write(lines[0])
	fp2.write(lines[1])
	for l in lat:
		for i in l:
			fp2.write(str(i)+' ')
		fp2.write('\n')
	fp2.write(lines[5])
	fp2.write(lines[6])
	fp2.write(lines[7])
	for site in newC:
		#print(str(site[0])+ ' '+str(site[1]) + ' ' + str(site[2]))
		fp2.write(str(round(float(site[0]),10)) + '  '+str(round(float(site[1]),10)) + '  ' + str(round(float(site[2]),10)))
		fp2.write('\n')



################################################################################################
def DFTU_POSCAR(filename,atom_position,check):
	#atom_position = int(atom_position)
	#print(atom_position)
    d = DFTU_POSCAR_PREP(filename,atom_position)
    f_open = open(filename,'r')
    lines = f_open.readlines()
    f_open.close()
    f_open = lines
    atom_type = f_open[5].split()
    atom_amount = f_open[6].split()
    atom_sum = [0]
    atom_sum = [float(atom_amount[0])]
    for i in np.arange(len(atom_type)):
        if i >0:
            atom_sum.append(atom_sum[i-1]+float(atom_amount[i]))

        if atom_sum[i] >= atom_position:
            element = atom_type[i-1]
            if i == 0:
                element = atom_type[i]
                lower_element = atom_position-1
            else:
                lower_element = int(atom_position-atom_sum[i-1]-1)
    
            higher_element= int(atom_sum[i]-atom_position)
            if atom_amount[i-1] != 1 and i!=0:
                if atom_amount[i-1] !=1:
                    atom_type.pop(i-1)
                    atom_amount.pop(i-1)
    
                if higher_element >0:
                    atom_type.insert(i-1,element)
                    atom_amount.insert(i-1,higher_element)
        
                atom_type.insert(i-1,element)
                atom_amount.insert(i-1,'1')
                species = i+1
                if lower_element > 0:
                    atom_type.insert(i-1,element)
                    atom_amount.insert(i-1,lower_element)
                    species = species+1
                    
                newline5 = ''
                for j in atom_type:
                    newline5 = newline5+str(j)+'   '
                    newline6 = ''
            
                for j in atom_amount:
                    newline6 = newline6+str(j)+'    '
            				#print(newline5)
            				#print(newline6)
                    newline5 = newline5+'\n'
                    newline6 = newline6+'\n'
                    f_open[5] = newline5
                    f_open[6] = newline6
                    break
    
            elif atom_amount[i] != 1 and i==0:
                if atom_amount[i] !=1:
                    atom_type.pop(i)
                    atom_amount.pop(i)
                
                if higher_element >0:
                    atom_type.insert(i,element)
                    atom_amount.insert(i,higher_element)
                
                atom_type.insert(i,element)
                atom_amount.insert(i,'1')
                species = i+1
                if lower_element > 0:
                    atom_type.insert(i,element)
                    atom_amount.insert(i,lower_element)
                    species = species+1
                
                newline5 = ''
                for j in atom_type:
                    newline5 = newline5+str(j)+'   '
                    newline6 = ''
                
                for j in atom_amount:
                    newline6 = newline6+str(j)+'    '
                
                newline5 = newline5+'\n'
                newline6 = newline6+'\n'
                f_open[5] = newline5
                f_open[6] = newline6
                break

    species = [species,atom_type,atom_amount]
    print(species)
    f_write = open('POSCAR_PREPPEDs','w')
    for i in f_open:
        f_write.write(i)
    f_write.close()
    sh.move('POSCAR_PREPPEDs','POSCAR')
    
    return species 

#############################################################################################################

def DFTU_POSCAR_PREP(filename,atom_position):
    if os.path.exists(filename) == False or os.path.getsize(filename) == 0:
        pass
    else:
        poscar = Structure.from_file(filename)
        species = poscar.species[int(atom_position-1)]
        print(species)
        coord = poscar.frac_coords[int(atom_position-1)]
        print(coord)
        poscar.insert(0,species,coord)
        poscar.pop(index=int(atom_position))
        poscar.to('poscar','POSCAR_PREPPED')
        print('SUCCESS!!')

#############################################################################################################

def DFTU_POTCAR(filename,potcar_location):
#input filename as the POSCAR file
#POTCAR_location is the directory where potcars stored under the format POTCAR_Xx is stored. Xx represents a element

	cwd = os.getcwd()
	os.chdir(potcar_location)
	filename = cwd+'/'+filename
	f_open = open(filename,'r').readlines()
	atom_type = f_open[5].split()
	atom_amount = f_open[6].split()
	if os.path.exists('POTCAR_temp') == True:
		os.remove('POTCAR_temp')
	loc = next(os.walk(potcar_location))[0]
	x_temp = next(os.walk(potcar_location))[2]
	x = []
	for i in x_temp:
		if i.find('POTCAR')>-1:
			x.append(i)
	potcar_atom_type = []
	for i in x:
		#print(i)
		temp = open(i,'r').readlines()
		potcar_atom_type.append((temp[0].split()))
	#print(potcar_atom_type)
	potcar = open('POTCAR_temp','w')
	for i in np.arange(len(atom_type)):
		#print('i = ',atom_type[i])
		for j in np.arange(len(potcar_atom_type)):
			for k in potcar_atom_type[j]:
				#print('Looking for = ',atom_type[j])
				#print(k)
				if k.find(atom_type[i])>-1:
					potcar_temp = open(x[j],'r').readlines()
					for line in potcar_temp:
						potcar.write(line)
	sh.move('POTCAR_temp',cwd+'/'+'POTCAR_temp')
	os.chdir(cwd)    



#################################################################################################
def DFTU_LDAU_tags(filename,species,U):
	print(os.getcwd())
	#species = DFTU_POSCAR(filename,species,0)
	species = get_species_list(filename)	
	LDAUL_line = ''
	for i in np.arange(len(species)):
		#print(species[1][i],' ',type(species[1][i]))
		orbital = Element(species[i]).block
		if orbital == 'd':
			LDAUL = 2
		elif orbital == 'p':
			LDAUL = 1
		elif orbital == 's':
			LDAUL = 0
		elif orbital == 'f':
			LDAUL = 3
		else:
			LDAUL = -1
		if i != 0:
			LDAUL = -1
		LDAUL_line = LDAUL_line+str(LDAUL)+' '
	LDAUU_line = ''
	LDAUJ_line = ''
	for i in np.arange(len(species)):
		if i == 0:
			LDAUU_line = LDAUU_line+str(U)+' '
			LDAUJ_line = LDAUJ_line+str(0)+' '
		else:
			LDAUU_line = LDAUU_line+str(0)+' '
			LDAUJ_line = LDAUJ_line+str(0)+' '
	tags = [LDAUU_line,LDAUJ_line,LDAUL_line]
	print(tags)
	return tags


#################################################################################################


def DFTU_Perturb_U(filename,species,U):
	DFTU_POSCAR('POSCAR',species,0)
	f = open(filename,'r').readlines()
	for i in f:
		if i.find('LDAUU') >-1:
			line = i.split()
			break
		line_number = line_number + 1
	for i in np.arange(len(line)):
		try:
			float(line[i])
			tag_start = i
			break
		except:
			print('LDAUU')
	newline = []
	for i in np.arange(tag_start,len(line)):
		if i == species:
			newU = float(U)
			newline.append(str(newU))
		else:
			newline.append(line[i])
	LDAUU_line = ''
	for i in newline:
		LDAUU_line = LDAUU_line+i+' '
	print(LDAUU_line)
	Replace_Incar('LDAUU',LDAUU_line)
	Replace_Incar('LDAUJ',LDAUU_line)
########################################################################################################


def DFTU_LR_tags():
	Replace_Incar('ISPIN','2')
	Replace_Incar('LCHARG','FALSE')
	Replace_Incar('LWAVE','FALSE')
	Replace_Incar('NUPDOWN','remove')
	Replace_Incar('MAGMOM','remove')
	Replace_Incar('LMAXMIX','6')
	Replace_Incar('LASPH','TRUE')
	Replace_Incar('ISYM','0')
	# DFTU tags
	Replace_Incar('LORBIT','11')
	Replace_Incar('LDAU','TRUE')
	Replace_Incar('LDAUTYPE','3')
	Replace_Incar('LDAUPRINT','2')
        

	# making sure ICHARGE is "fresh"
	Replace_Incar('ICHARG','remove')

#########################################################################################################################

def DFTU_SCF_RUN(file_location,potcar_location,atom_position,U_values):
    filenames = []
    keep = ['INCAR','POSCAR','CONTCAR','KPOINTS','runjob','CHGCAR','WAVECAR','vdw_kernel.bindat']
    remove = ['OUTCAR','OSZICAR','vasprun.xml']
    cwd = os.getcwd()
    for i in U_values:
        filename = 'calc_'+str(i)
        filenames.append(filename)
        if os.path.exists(filename) == False:
            os.mkdir(filename)
        for j in keep:
            if os.path.exists(file_location+'/'+j) == True:
                if j == 'CONTCAR':
                    try:
                        if os.path.exists(filename+'/CONTCAR') == False:
                            sh.copy(j,filename+'/'+j)
                            sh.move(filename+'/CONTCAR',filename+'/'+'POSCAR')
                        else:
                            sh.move(filename+'/CONTCAR',filename+'/'+'POSCAR')
                    except:
                            sh.copy('POSCAR',filename+'/POSCAR')
                else:
                    sh.copy(file_location+'/'+j,filename+'/'+j)
                    print(j, ' HAS BEEN COPIED OVER FROM ',file_location)
            else:
                print(j,' IS NOT PRESENT AND WILL NOT BE COPIED')
        os.chdir(filename)
        for j in remove:
            if os.path.exists(j) == True:
                os.remove(j)
        species = DFTU_POSCAR('POSCAR',atom_position,1)
        sh.move('POSCAR_temp','POSCAR')
        DFTU_POTCAR('POSCAR',potcar_location)
        sh.move('POTCAR_temp','POTCAR')
        tags = DFTU_LDAU_tags('POSCAR',species[0],i)
        print(tags)
        Replace_Incar('LASPH','TRUE')
        Replace_Incar('LMAXMIX','6')
        Replace_Incar('LCHARG','TRUE')
        Replace_Incar('LORBIT','11')
        Replace_Incar('ISYM','0')
        Replace_Incar('ISPIN','2')
        Replace_Incar('NEDOS','2000')
        Replace_Incar('LDAU','TRUE')
        Replace_Incar('LASPH','TRUE')
        Replace_Incar('LDAUTYPE','3')
        Replace_Incar('LDAUPRINT','2')
        Replace_Incar('LDAUU',tags[0])
        Replace_Incar('LDAUJ',tags[0])
        Replace_Incar('LDAUL',tags[1])
        os.chdir(cwd)
    return str(filename)

########################################################################################################


def DFTU_LR_GRAPH(Bare,Inter,U0,directory,Bare_trend=[],Inter_trend=[]):
    cwd = os.getcwd()
    if len(Bare_trend)>0 and len(Inter_trend) >0:
        x_bare_trend = Bare_trend[0]
        y_bare_trend = Bare_trend[1]
        x_inter_trend = Inter_trend[0]
        y_inter_trend = Inter_trend[1]
        plt.plot(x_bare_trend,y_bare_trend,'r')
        plt.plot(x_inter_trend,y_inter_trend,'r')

    x_bare = Bare[0]
    y_bare = Bare[1]
    x_inter = Inter[0]
    y_inter = Inter[1]
    print('BARE:')
    for i in range(len(x_bare)):
        print(x_bare[i],'\t',y_bare[i])

    print('\n')

    for i in range(len(x_bare)):
        print(x_inter[i],'\t',y_inter[i])

    plt.scatter(x_bare,y_bare)
    plt.scatter(x_inter,y_inter,marker='*')
    plt.xlabel('ALPHA (eV)')
    plt.ylabel('$\Delta$ n : d-orbital electrons')
    title_text = 'Linear Response U: U0 = '+str(U0)
    plt.title(title_text)

    if len(Bare_trend)>0 and len(Inter_trend)>0:
        plt.legend(['Bare Trendline','Interacting Trendline','Bare','Interacting'])
    else:
        plt.legend(['BARE','INTERACTING'])

    filename = str(U0)+'_LR_GRAPH.png'
    plt.savefig(filename)
    plt.close()
    sh.move(filename,directory+'/'+filename)
    os.chdir(directory)

    if os.path.exists(cwd+'/GRAPHS') == False:
        os.mkdir(cwd+'/GRAPHS')

    sh.move(filename,cwd+'/GRAPHS/'+filename)
    os.chdir(cwd)
###################################################################################################

def DFTU_SCF_U_GRAPH(Uin,Uout,directory):
	cwd = os.getcwd()
	plt.scatter(Uin,Uout)
	plt.xlabel('U_in (eV)')
	plt.ylabel('U_out (eV)')
	title_text = 'SCF LR  Hubbard U'
	plt.title(title_text)
	filename = 'SCF_U_GRAPH.png'
	plt.savefig(filename)
	plt.close()
	if os.path.exists('GRAPHS') == False:
		os.mkdir('GRAPHS')
	sh.move(filename,'GRAPHS/'+filename)
	os.chdir(cwd)




	 
######################################################################################################

def get_job_dir(usrname):
	slurm_job_command = 'squeue -u '+usrname+' -o "%Z"'
	job_dir_list = list(os.popen(slurm_job_command))
	for i in np.arange(len(job_dir_list)):
		job_dir_list[i] = job_dir_list[i][0:(len(job_dir_list[i])-1)]  
	return job_dir_list

#####################################################################################################

# code taken from : https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


######################################################################################################

def run_vasp(adir,relax):
    # This function excutes vasp in a director "adir" it will continue to run until the calculation is converged or the max_interation limit is reached. This script will only allow a job to resubmit up to 3 times, copying CONTCAR to POSCAR between runs.


    cwd = os.getcwd()
    ml_command = 'module load intel/2019.1.144 openmpi/4.0.1'
    vasp_command = 'mpirun  /home/joshuapaul/vasp_10-23-19_5.4.4/bin/vasp_std > job.log'
    os.chdir(adir)
    converged = False
    count = 0

    while converged == False:
        os.system(ml_command) 
        os.system(vasp_command)
        count = count + 1

        if relax == True:
            if os.path.exists('vasprun.xml'):
                vasprun = Vasprun('vasprun.xml')
                if vasprun.converged == True:
                    converged = True
                    print(adir + ' completed')

                if converged == False:
                    sh.copy('OSZICAR','OSZICAR.'+str(count))
                    sh.copy('POSCAR','POSCAR.'+str(count))
                    sh.copy('CONTCAR','POSCAR')
                    print('calculation restarting . . . copying CONTCAR to POSCAR')
        else:
            if os.path.exists('vasprun.xml') == True:
                vasprun = Vasprun('vasprun.xml')
                if vasprun.converged_electronic == True:
                    converged = True
        
        if count == 2:
            converged = True
            break
        else:
            pass

    os.chdir(cwd)
    return converged

########################################################################################################
def get_POTCAR(file_directory):
    cwd = os.getcwd()
    os.chdir(file_directory)
    poscar_path = file_directory+'/POSCAR'
    species = get_species_list(poscar_path)
    potcar = Potcar(species)
    potcar.write_file('POTCAR')
    return potcar 
    

def get_species_list(filename):
    f_open = open(filename,'r').readlines()
    atom_type = f_open[5].split()
    return atom_type
########################################################################################################

def run_vasp2(adir,relax):
    # This function excutes vasp in a director "adir" it will continue to run until the calculation is converged or the max_interation limit is reached. This script will only allow a job to resubmit up to 3 times, copying CONTCAR to POSCAR between runs.


    cwd = os.getcwd()
    
    vasp_command = 'srun --mpi=pmix_v3 /home/joshuapaul/vasp_10-23-19_5.4.4/bin/vasp_std > job.log'
    os.chdir(adir)
    converged = False
    count = 0

    while converged == False: 
        os.system(vasp_command)
        count = count + 1

        if relax == True:
            if os.path.exists('OUTCAR'):
                with open('OUTCAR') as out:
                    lines = out.readlines()
                    for line in lines:
                        if 'reached required accuracy' in line:
                            converged = True
                            print('calcualtion reached required accuracy')
                            break
                if converged == False:
                    sh.copy('OSZICAR','OSZICAR.'+str(count))
                    sh.copy('POSCAR','POSCAR.'+str(count))
                    sh.copy('CONTCAR','POSCAR')
                    print('calculation restarting . . . copying CONTCAR to POSCAR')
        else:
            try:
                if os.path.exists('vasprun.xml') == True:
                    vasprun = Vasprun('vasprun.xml')
                    if vasprun.converged_electronic == True:
                        converged = True
            except:
                converged = True

        if count == 2:
            converged = False
            break
        else:
            pass

    os.chdir(cwd)
    return converged

########################################################################################################

def nw_memory(filename):
    with open(filename,'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.find('maximum total bytes') > -1:
                memory = line.split()[-1]
                return memory

########################################################################################################

def run_nwchem(adir):
    # This function excutes vasp in a director "adir" it will continue to run until the calculation is converged or the max_interation limit is reached. This script will only allow a job to resubmit up to 3 times, copying CONTCAR to POSCAR between runs.

    cwd = os.getcwd()
    converged = False
    nwchem_command = 'srun --mpi=pmix_v3 nwchem input.nw > output.nw'
    os.chdir(adir)
    while converged == False:
        os.system(nwchem_command)
        converged = True
        with open(adir+'/output.nw','r') as output:
            lines = output.readlines()
            for line in lines:
                if line.find('no. of electrons and multiplicity not compatible') > -1:
                    mult = up_mult_nwchem(adir)
                    converged = False
                    if mult == 4:
                        converged = True
                        break
        
    os.chdir(cwd)
    return converged

########################################################################################################
def occ(x,y,zero):
    loc = np.where(x<=zero)[0]
    x = x[loc]
    y = y[loc]
    summation = np.sum(y)*2-y[0]-y[-1]
    trap_sum = summation*np.abs(x[-1]-x[0])/(2*len(x))
    return trap_sum

#######################################################################################################

def get_spd_occupation(filename):
    # inputs file path to a vasprun.xml
    vasprun = Vasprun(filename)
    sites = [i.specie for i in vasprun.final_structure.sites]
    cdos = vasprun.complete_dos
    result = []
    for site in sites:
        spd = cdos.get_element_spd_dos(site)
        spdkeys = list(spd.keys())
        for orbital in spdkeys:
            total_density = spd[orbital].densities
            densitykeys = list(total_density.keys())
            spin_density = []
            for spin in densitykeys:
                spin_density.append(occ(cdos.energies,total_density[spin],zero=vasprun.efermi))
            result.append(spin_density)
    return result

########################################################################################################

def up_mult_nwchem(directory):
    filename = directory+'/input.nw'
    with open(filename,'r') as nwinput:
        lines = nwinput.readlines()
        for index in range(len(lines)):
            if lines[index].find('mult') == True:      
                mult = int(lines[index].split()[-1])+1
                lines[index] = ' mult '+str(mult)+'\n'

    with open(filename+'temp','w') as nwinput_write:
        for line in lines:
            nwinput_write.write(line)
    sh.move(filename+'temp',filename)
    return mult
##################################################################################
def nw_hlgap(path):
    #returns [up_homo/lumo, down_homo/lumo] in eV
    conv_hart_eV = 27.114
    orb_line = []
    virt_line = []
    with open(path+'/output.nw','r') as output:
        lines = output.readlines()[::-1]
        for i in range(len(lines)):
            if lines[i] == ' orbital energies:\n':
                if len(lines[i+1].split()) < 3:
                    orb1 = float(lines[i-1].split()[0])
                    orb2 = float(lines[i-2].split()[-3])
                    orb_line = np.array([orb1,orb2])
                else:
                    orb1 = float(lines[i-1].split()[0])
                    orb2 = float(lines[i-1].split()[-3])
                    orb_line = np.array([orb1,orb2])
                break
        for i in range(len(lines)):
            if lines[i] == ' virtual orbital energies:\n':
                virt1 = float(lines[i-1].split()[0])
                virt2 = float(lines[i-1].split()[-3])
                virt_line = np.array([virt1,virt2])
                break
    hl = np.min(virt_line) - np.max(orb_line)
    hl = hl*conv_hart_eV
    return [orb_line,virt_line,hl]
###################################################################################
def nw_converged(output_file_path):
    try:
        converged = False
        with open(output_file_path+'/output.nw','r') as output:
            lines = output.readlines()[::-1]
            for i in range(len(lines)):
                if lines[i].find('allocation') > -1:
                    converged = True
    except:
        converged = None
        pass
    return converged

##########################################################################################

def nw_final_structure(output_file_path,string_prefix):
    files = next(os.walk(output_file_path))[2]
    strucs = []
    step_numbers = []
    for filename in files:
        try:
            filename = filename.decode()
        except:
            pass
        if filename.find('.xyz') > -1 and filename.find(string_prefix) > -1:
            strucs.append(filename)
    for struc in strucs:
        step_numbers.append(float(struc.split('-')[-1].split('.')[0]))
    if len(step_numbers) > 0:
        max_iter = max(step_numbers)
        loc = [i for i in range(len(step_numbers)) if step_numbers[i] == max_iter][0]
        final_struc_name = strucs[loc]
        try:
            final_struc_name = final_struc_name.decode()
        except:
            pass
        try:
            final_struc = XYZ.from_file(output_file_path+'/'+final_struc_name).molecule
        except:
            return [None, None]
        return [final_struc,final_struc_name]
    else:
        return [None,None]
#######################################################################################

def pymatgen_to_ase(pymatgen_molecule):
    pymatgen_molecule.to('xyz','temp.xyz')
    ase_molecule = io.read('temp.xyz')
    os.remove('temp.xyz')
    return ase_molecule
def ase_to_pymatgen(ase_molecule):
    ase_molecule.write('temp.xyz',format='xyz')
    pymatgen_moelcule = XYZ.from_file('temp.xyz').molecule
    os.remove('temp.xyz')
    return pymatgen_moelcule

def nearest_neighbors(ase_molecule,n):
#    pymatgen_molecule = ase_to_pymatgen(ase_molecule)
#    distances = [pymatgen_molecule.get_distance(0,i) for i in range(len(pymatgen_molecule.sites))]
    distances = np.array(ase_molecule.get_distances(0,range(len(ase_molecule))))
    loc = np.argsort(distances)
    mole = ase_molecule[loc[0:n]]
    return mole

def normalize_descriptors(descriptors):
    cm_normal = []
    cm_maxes = []
    for i in descriptors:
        cm_maxes.append(np.max(i))
    for index in range(len(descriptors)):
        cm_normal.append(descriptors[index]/cm_maxes[index])

def dask_workers(cores,mem_per_core,njobs,burst=True,create_workers=True,address=None):
    # input mem in gb/core
    if create_workers == True:
        if burst == True:
            cluster_job = SLURMCluster(cores=1,
                memory=str(round(cores*mem_per_core))+'GB',
                project='hennig',
                walltime='72:00:00',
                processes=1,
                job_extra=['--cpus-per-task=1',
                '--ntasks='+str(cores),
                '--nodes='+str(int(np.ceil(cores/32))),
                '--qos=hennig-b'])
        else:
            cluster_job = SLURMCluster(cores=1,
                memory=str(round(cores*mem_per_core))+'GB',
                project='hennig',
                walltime='72:00:00',
                processes=1,
                job_extra=['--nodes='+str(int(np.ceil(cores/32))),
                    '--cpus-per-task=1',
                    '--ntasks='+str(cores)])
        # a dask "client" manages all the workers
        cluster_job.scale(jobs=njobs)
        client = Client(cluster_job)
    else:
        client = Client(address)
        cluster_job = client.cluster
        pass
    return [client,cluster_job]

def dask_workers2(cores,mem_per_core,njobs,burst=True,create_workers=True,address=None):
    # input mem in gb/core
    if create_workers == True:
        if burst == True:
            cluster_job = SLURMCluster(cores=cores,
                memory=str(round(cores*mem_per_core))+'GB',
                project='hennig',
                walltime='72:00:00',
                processes=1,
                job_extra=['--cpus-per-task=1',
                '--ntasks='+str(cores),
                '--nodes='+str(int(np.ceil(cores/32))),
                '--qos=hennig-b'])
        else:
            cluster_job = SLURMCluster(cores=cores,
                memory=str(round(cores*mem_per_core))+'GB',
                project='hennig',
                walltime='72:00:00',
                processes=1,
                job_extra=['--nodes='+str(int(np.ceil(cores/32))),
                    '--cpus-per-task=1',
                    '--ntasks='+str(cores)])
        # a dask "client" manages all the workers
        cluster_job.scale(jobs=njobs)
        client = Client(cluster_job)
    else:
        client = Client(address)
        cluster_job = client.cluster
        pass
    return [client,cluster_job]
###########################################################################
def Pending_jobs(futures):
# Pending_jobs returns the number of jobs in a futures list that return 'pending' from future.status
# returns : [ number of pending futures, number of futures which dont report pending]
    pending = 0
    completed = 0
    for future in futures:
        if future.status == 'pending':
            pending += 1
        else:
            completed += 1
    return [pending, completed]

##########################################################################
########################################################################################################

def run_vasp6(adir,relax):
    # This function excutes vasp in a director "adir" it will continue to run until the calculation is converged or the max_interation limit is reached. This script will only allow a job to resubmit up to 3 times, copying CONTCAR to POSCAR between runs.

    cwd = os.getcwd()
    
    vasp_command = 'srun --mpi=pmix_v3 /home/joshuapaul/vasp_10-23-19_5.4.4/bin/vasp_std > job.log'
    os.chdir(adir)
    converged = False
    count = 0

    while converged == False: 
        os.system(vasp_command)
        count = count + 1

        if relax == True:
            if os.path.exists('OUTCAR'):
                with open('OUTCAR') as out:
                    lines = out.readlines()
                    for line in lines:
                        if 'reached required accuracy' in line:
                            converged = True
                            print('calcualtion reached required accuracy')
                            break
                if converged == False:
                    sh.copy('OSZICAR','OSZICAR.'+str(count))
                    sh.copy('POSCAR','POSCAR.'+str(count))
                    sh.copy('CONTCAR','POSCAR')
                    print('calculation restarting . . . copying CONTCAR to POSCAR')
        else:
            try:
                if os.path.exists('vasprun.xml') == True:
                    vasprun = Vasprun('vasprun.xml')
                    if vasprun.converged_electronic == True:
                        converged = True
            except:
                converged = True

        if count == 2:
            converged = False
            break
        else:
            pass

    os.chdir(cwd)
    return converged


##################################################################################

def create_cosmo(xyz_path, functional, basis, scrf, chg, spin, radii_df = [], opt=True):
    chg = str(chg)
    spin = str(spin)
    xyz = XYZ.from_file(xyz_path)
    coords = xyz.molecule.cart_coords
    species = [elem.name for elem in xyz.molecule.species]
    formula = xyz.molecule.formula
    file_name = xyz_path[:-4].split('/')[-1]
    chk = '%chk=' + file_name + '.chk\n'
    cosmo = file_name + '_' + functional +'.cosmo\n'
    #input_name = file_name + '_' + 
    preamble = '# ' + functional + '/' + basis + ' scrf=(' + scrf + ',read)'+' Integral=UltraFine'
    if opt == True:
        preamble = preamble + ' opt\n'
    else:
        preamble = preamble + '\n'
     
    link = '--Link1--\n'
    coord_lines = []
    file = [chk, preamble, '\n', file_name + '\n', '\n', chg + ' ' + spin, '\n']
    
### if a radii dictionary is found, find atoms to change radii
    if len(radii_df) > 0:
        radii_lines = []
        radii = []
        for atom in species: 
            print('ATOM : ',atom)
            print('CHG  : ',chg)
            radius = radii_df[radii_df['Ion'] == atom]['Radius']
            if len(radius) > 1:
                candidates = radii_df[radii_df['Ion'] == atom] 
                if len(species) == 1:
                    radius = candidates[candidates['Charge'] == float(chg)]['Radius']
            radius = radius.values[0]
            print('RADIUS :', type(radius))
            radii.append(radius)
        if len(radii) > 0 :
            for i, radius in enumerate(radii):
                radii_line = str(i+int(1)) + ' ' + str(radius) + ' 1.0\n'
                radii_lines.append(radii_line)

# write the coordinates out
    for coord_set in coords:
        coord_line = [coord_set[0],coord_set[1],coord_set[2]]
        coord_lines.append(coord_line)
    coord_lines = np.array([['{0:12f}'.format(i)+'\t' for i in c] for c in coord_lines])
    
    for index, value in enumerate(coord_lines):
        line = species[index] + ' ' + value[0] + value[1] + value[2]
        print(line)
        line = line + '\n'
        file.append(line)
    file.append('\n')

### write the modifysph tag
    if len(radii_df) > 0 : 
        if len(radii_lines) > 0 : 
            file.append('modifysph\n')
            file.append('\n')
            for radii_line in radii_lines:
                file.append(radii_line)
    file.append('\n')
    
    file.append(link)
    preamble = '# ' + functional + '/' + \
                     basis + \
                     ' scrf=(' + 'cosmors' + ',read)'+ \
                     ' geom=check' + \
                     ' guess=check' + \
                     ' scf=maxcycle=5000' +\
                     ' opt Integral=UltraFine\n'
                     

###########################################################################
# you can use library radii values using the lines that are commented below
#    radii = 'radii=bondi\n'
 
    #cosmo_file = [chk, '\n', preamble, '\n', 
    #              file_name + '\n', '\n', chg + ' ' + spin + '\n\n' + cosmo + '\n', 
    #              '\n',radii]

###########################################################################

    cosmo_file = [chk, preamble, '\n', 
                  file_name + '\n', '\n', chg + ' ' + spin + '\n\n']
    file.extend(cosmo_file)
    
    if len(radii_df) > 0 : 
        file.append('modifysph\n')
        file.append('\n')
        for radii_line in radii_lines:
            file.append(radii_line)
    file.append('\n')
    file.append(cosmo)
    file.append('\n')
    return file

#################################################################################################

def create_gau(xyz_path, functional, basis, chg, spin, radii_df = []):
    chg = str(chg)
    spin = str(spin)
    xyz = XYZ.from_file(xyz_path)
    coords = xyz.molecule.cart_coords
    species = [elem.name for elem in xyz.molecule.species]
    formula = xyz.molecule.formula
    file_name = xyz_path[:-4].split('/')[-1]
    chk = '%chk=' + file_name + '.chk\n'
    preamble = '# ' + functional + '/' + \
                basis + \
                ' opt Integral=UltraFine\n' 
    if len(radii_df) > 0:
        radii_lines = []
        radii = []
        for atom in species: 
            print(atom)
            radius = radii_df[radii_df['Ion'] == atom]['Radius']
            if len(radius) > 1:
                radius = radii_df[(radii_df['Ion'] == atom) & (radii_df['Charge'] == chg)]['Radius']
            radii.append(radius)
        if len(radii) > 0 and len(radii[0]) > 0:
            print(radii)
            for i, radius in enumerate(radii):
                radii_line = str(i+int(1)) + ' ' + str(radius.iloc[0]) + ' 1.0\n'
                radii_lines.append(radii_line)
    coord_lines = []
    file = [chk, preamble, '\n', file_name + '\n', '\n', chg + ' ' + spin, '\n']
    for coord_set in coords:
        coord_line = [str(round(coord_set[0],5)), 
                    str(round(coord_set[1],5)),
                    str(round(coord_set[2],5))]
        coord_lines.append(coord_line)
    
    coord_lines = np.array([[i+'\t' for i in c] for c in coord_lines])
    
    for index, value in enumerate(coords):
        line = species[index] + ' '
        for j in value:
            line = line + str(j) + ' '
        line = line + '\n'
        file.append(line)
    file.append('\n')
    file.append('modifysph\n')
    if len(radii_lines) > 0 : 
        for radii_line in radii_lines:
            file.append(radii_line)
    file.append('\n')
    
    return file


#############################################################################################

def dmol_get_xyz(fname):
    with open(fname, 'r') as input_file:
        lines = open(fname, 'r').readlines()

        for i, line in enumerate(lines):
            if line.find('$coord') > -1:
                begin = i
                break

        for i, line in enumerate(lines):
            if line.find('$end') > -1:
                end = i
                break

        coords = lines[begin+1:end]
        natoms = len(coords)
    write_name = fname.split('.')[0] + '.xyz'
    with open(write_name, 'w') as write_file:
        write_file.write(str(natoms) + '\n')
        write_file.write('\n')
        for coord in coords:
            write_file.write(coord)
        write_file.write('\n')
    return coords


#########################################################

def get_dmol(fname):
    output = []
    with open(fname,'r') as output_file:
        lines = output_file.readlines()
        for index,line in enumerate(lines):
            value = np.NAN
            if line.find('DMol3/COSMO Results') > -1:
                start_loc = index
            if line.find('End Computing SCF Energy/Gradient') > -1:
                end_loc = index
        for index in range(start_loc,end_loc):
            line = lines[index]
            if line.find('=') > -1:
                value = float(line.split('=')[-1].split(' ')[-1])
                output.append(value)
        output = {'Area':output[6],
                    'Volume':output[7],
                    'Solvation Energy':output[0],
                    'Dielectric_Energy':output[8],
                    'q':output[4],
                    'q"':output[5]}

    return output
#####################################################

def run_NLN(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    os.system('ml intel/2020.0.166 openmpi && mpirun /blue/hennig/ericfonseca/NASA/VASPsol/Redhat/NLN/bin/vasp_std')
    os.chdir(cwd)
    return

####################################
def plot_locpot(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    lpb = Locpot.from_file('LOCPOT')
    s = Structure.from_file('POSCAR')
    y = lpb.get_average_along_axis(3)
    x = np.linspace(0, s.lattice.c, len(y))
    fig = plt.figure(dpi=300)
    plt.plot(x,y, label='LPB')
    os.chdir(cwd)
    return fig


#def soap_from_xyz(filename, rcut = 5, nmax=3, lmax = 4):
#    xyz = io.read(filename)
#    species = set()
#    species.update(xyz.get_chemical_symbols())
#    soap = SOAP(species=species, periodic=False, rcut=rcut, nmax=nmax, lmax=lmax, average='outer', sparse=False)
#    vect = soap.create(xyz, positions=[0])
#    return vect

