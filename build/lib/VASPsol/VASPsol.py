import dask
from ase import io
from scipy.interpolate import griddata
from matplotlib import ticker, cm
from pymatgen.core.surface import SlabGenerator
import subprocess 
import shutil as sh
from custom import custom as ef
import os
import pandas as pd
from pymatgen.io.vasp.outputs import Oszicar
from pymatgen.io.vasp.outputs import Vasprun, Chgcar, Locpot, Outcar
from pymatgen.analysis.surface_analysis import WorkFunctionAnalyzer
from pymatgen.core import Structure
from pymatgen.core import surface
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

def generate_surface(struc, vector, min_slab_dim, min_vac, **kwargs):
    slab = SlabGenerator(struc, vector, min_slab_dim, min_vac, center_slab=True, **kwargs).get_slab()
    slab.to('POSCAR','POSCAR')
    return slab


def solvent_dataframe():
    try:
        df = pd.read_csv('/home/ericfonseca/Solvents.csv')
    except:
        df = pd.read_csv('~/Desktop/Solvents.csv')
    return df

def solvent_properties(solvent_name):
    properties = {}
    df = solvent_dataframe()
    for col in df.columns[1::]:
        properties[col] = df[df['Solvent'] == solvent_name][col].to_numpy()[0]
    return properties

def add_solvent(directory, solvent='water', epsilon = 78.4, Ionic_strength = 0, calc_type='linear', NSW=100, dict_of_additional_tags={}):
    print('ADDING PROPERTIES : ' + solvent)
    os.chdir(directory)
    ef.Replace_Incar('LSOL','TRUE')
    ef.Replace_Incar('NSW', NSW)
    if Ionic_strength > 0:
        Ionic_strength = 3.04 / Ionic_strength**(1/2)
        ef.Replace_Incar('LAMBDA_D_K',Ionic_strength)
    dict_keys = list(dict_of_additional_tags.keys())
    for key in dict_keys:
        ef.Replace_Incar(key, dict_of_additional_tags[key])
    df = solvent_dataframe()
    if calc_type == 'linear':
        print(df[df['Solvent'] == solvent]['EB_K'].to_numpy())
        ef.Replace_Incar('EB_K', df[df['Solvent'] == solvent]['EB_K'].to_numpy()[0])
    elif calc_type == 'non-linear':
        ef.Replace_Incar('LNLSOL','TRUE')
        for col in df.columns[1::]:
            ef.Replace_Incar(col, df[df['Solvent'] == solvent][col].to_numpy()[0])
    return

def remove_Vaspsol_tags(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    ef.Replace_Incar('LSOL','remove')
    ef.Replace_Incar('LAMBDA_D_K','remove')
    ef.Replace_Incar('NC_K','remove')
    ef.Replace_Incar('LRHOION','remove')
    ef.Replace_Incar('LRHOB','remove')
    ef.Replace_Incar('EB_K','remove')
    ef.Replace_Incar('P0_K','remove')
    ef.Replace_Incar('P0_VAC','remove')
    ef.Replace_Incar('EFINITY_K','remove')
    ef.Replace_Incar('TAU','remove')
    ef.Replace_Incar('S_Z_START','remove')
    ef.Replace_Incar('S_Z_END','remove')
    ef.Replace_Incar('GDC_SIGMA','remove')
    os.chdir(cwd)
    return


def calculate_vacuum(directory, restart=False, NSW=100, dict_of_additional_tags={}):
    cwd = os.getcwd()
    directory = os.path.abspath(directory)
    try:
        if directory[-1] == '/':
            directory = directory[0:-1]
        vac_dir = directory + '/VAC'
        if restart == True:
            try:
                os.remove('VAC/vasprun.xml')
            except:
                pass
        #os.chdir(directory)
        print('initializing vacuum run')
        initialize_vacuum(directory, dict_of_additional_tags=dict_of_additional_tags)
        print('STARTING VASP')
        run(vac_dir)
        vasprun = read_vasprun(vac_dir)
        os.chdir(cwd)
        return vasprun
    except:
        os.chdir(cwd)
        return 'Failed' 

def copy_vacuum_files(directory, copy_to_dir,  restart=False):
    vac_dir = directory + '/VAC'
    vac_dir = os.path.abspath(vac_dir)
    solv_dir = copy_to_dir
    solv_dir = os.path.abspath(solv_dir)
    keep_files = ['WAVECAR','POSCAR','CONTCAR','KPOINTS','POTCAR','INCAR']
    vac_files = next(os.walk(vac_dir))[2]
    for fname in vac_files:
        if fname in keep_files:
            try:
                if restart == True:
                    pass
                else:
                    fpath_vac = vac_dir + '/' + fname
                    fpath_solv = solv_dir + '/' + fname
                    sh.copy(fpath_vac, fpath_solv)
                    print('COPIED ', fname, ' from VAC')
            except:
                print('FAILED TO COPY FROM VAC : ',fname, ' USING INITIAL DIR INSTEAD')
                try:
                    fpath_vac = directory + '/' + fname
                    fpath_solv = solv_dir + '/' + fname
                    sh.copy(fpath_vac, fpath_solv)
                    print('COPIED ', fname, ' from INIT DIR')
                except:
                    print('Completely failed to find ', fname)
                

def copy_directory_files(directory, solvent='water', restart=False):
    directory = os.path.abspath(directory)
    if directory[-1] == '/':
        directory = directory[0:-1]
    solv_dir = directory + '/' + solvent
    solv_dir = os.path.abspath(solv_dir)
    keep_files = ['WAVECAR','POSCAR','CONTCAR','KPOINTS','POTCAR','INCAR']
    dir_files = next(os.walk(directory))[2]
    for fname in dir_files:
        if fname in keep_files:
            try: 
                if restart == True:
                    pass
                else:
                    fpath_dir = directory + '/' + fname
                    fpath_solv = solv_dir + '/' + fname
                    sh.copy(fpath_dir, fpath_solv)
                    print('GRABBING DUMMY VERSION : ', fname)
            except:
                print('FAILED TO COPY ', fname)


def initialize_solvent(directory, solvent='water', epsilon=78.4, Ionic_strength=1, calc_type='linear', NSW=100, dict_of_additional_tags={}):
    directory = os.path.abspath(directory)
    if directory[-1] == '/':
        directory = directory[0:-1]
    cwd = os.getcwd()
    vac_dir = directory + '/VAC'
    print(vac_dir)
    solv_dir = directory + '/' + solvent
    solv_dir = os.path.abspath(solv_dir)
    print(solv_dir)
    if os.path.exists(solv_dir) == False:
        os.mkdir(solv_dir)

    if os.path.exists(solv_dir+'/CONTCAR'):
        pass 
    else:
        if os.path.exists(vac_dir + '/vasprun.xml'):
            try:
                vac_vasprun = read_vasprun(vac_dir)
                print(vac_vasprun.converged)
                if vac_vasprun.converged == True:
                    copy_vacuum_files(directory, solv_dir)
                else:
                    copy_directory_files(directory, solvent=solvent)
            except:
                copy_directory_files(directory, solvent=solvent)
        else:
            copy_directory_files(directory, solvent=solvent)

    if os.path.exists(solv_dir + '/CONTCAR'):
        sh.copy(solv_dir + '/CONTCAR', solv_dir + '/POSCAR')

    add_solvent(solv_dir, solvent=solvent, calc_type=calc_type, NSW=NSW, dict_of_additional_tags=dict_of_additional_tags)
    os.chdir(cwd)
    return

def check_convergence(directory):
    directory = os.path.abspath(directory)
    if directory[-1] == '/':
        directory = directory[0:-1]
    try:
        vasprun = read_vasprun(directory)
        if vasprun.converged == True:
            return True
        else:
            return False
    except:
        return False
    
def initialize_vacuum(directory, dict_of_additional_tags={}):
    cwd = os.getcwd()
    directory = os.path.abspath(directory)
    if directory[-1] == '/':
        directory = directory[0:-1]
    vac_dir = directory + '/VAC'
    if os.path.exists(vac_dir):
        convergence = check_convergence(vac_dir)
        if convergence:
            pass
        else:
            copy_directory_files(directory, solvent='VAC')
    else:
        os.mkdir(vac_dir)
        copy_directory_files(directory, solvent='VAC')
    if os.path.exists(vac_dir + '/CONTCAR'):
        sh.copy(vac_dir + '/CONTCAR', vac_dir + '/POSCAR')
    os.chdir(vac_dir)
    print(vac_dir)
    remove_Vaspsol_tags(vac_dir)
    dict_keys = list(dict_of_additional_tags.keys())
    for key in dict_keys:
        ef.Replace_Incar(key, dict_of_additional_tags[key])
    os.chdir(cwd)
    return

def initialize_gdc(directory, dz=5, epsilon=78.4, Ionic_strength=1, NSW=100, restart = False, dict_of_additional_tags = {}):
    directory = os.path.abspath(directory)
    if directory[-1] == '/':
        directory = directory[0:-1]
    cwd = os.getcwd()
    vac_dir = directory + '/VAC'
    vac_dir = os.path.abspath(vac_dir)
    print(vac_dir)
    solvent = 'dz_' + str(dz) + '_epsilson_' + str(epsilon) + '_Ionic_' + str(Ionic_strength)
    tags = list(dict_of_additional_tags.keys())
    for tag in tags:
        solvent = solvent+'_'+tag+'_'+str(dict_of_additional_tags[tag])
    solv_dir = directory + '/' + solvent
    solv_dir = os.path.abspath(solv_dir)
    print(solv_dir)

    if os.path.exists(solv_dir) == False:
        os.mkdir(solv_dir)
    if restart == True:
        pass
    else:
        try:
            if os.path.exists(vac_dir + '/vasprun.xml'):
                print('FOUND a VACUUM RUN')
                vac_vasprun = read_vasprun(vac_dir)
                print(vac_vasprun.converged)
                if vac_vasprun.converged == True:
                    copy_vacuum_files(directory, solv_dir, restart=restart)
                else:
                    print('vasprun converged == False')
                    copy_directory_files(directory, solvent=solvent)
            else:
                copy_directory_files(directory, solvent=solvent)
        except:
            print('FOUND ERROR IN VAC VASPRUN, COPYING FILES ANYWAY')
            copy_vacuum_files(directory, solv_dir, restart=restart)

    if os.path.exists(solv_dir + '/CONTCAR'):
        sh.copy(solv_dir + '/CONTCAR', solv_dir + '/POSCAR')
    os.chdir(solv_dir)
    add_gdc2(solv_dir, dz=dz, epsilon=epsilon, Ionic_strength=Ionic_strength, NSW=NSW, dict_of_additional_tags = dict_of_additional_tags)
    #add_solvent(solv_dir, solvent=solvent, calc_type=calc_type, NSW=NSW)
    os.chdir(cwd)
    return solv_dir

def initialize_solvent2(directory, solvent='water', epsilon=78.4, calc_type='linear', Ionic_strength=0, NSW=100, restart = False, dict_of_additional_tags = {}):
    directory = os.path.abspath(directory)
    if solvent != 'water':
        try:
            props = solvent_properties(solvent)
            epsilon = props['EB_K']
        except:
            pass
    
    if directory[-1] == '/':
        directory = directory[0:-1]
    cwd = os.getcwd()
    vac_dir = directory + '/VAC'
    vac_dir = os.path.abspath(vac_dir)
    print(vac_dir)
    solv_dir = 'Solvent_' + str(solvent) + '_epsilson_' + str(epsilon) + '_Ionic_' + str(Ionic_strength) + '_calc_' + str(calc_type)
    tags = list(dict_of_additional_tags.keys())
    for tag in tags:
        solv_dir = solv_dir+'_'+tag+'_'+str(dict_of_additional_tags[tag])
    solv_dir = os.path.abspath(solv_dir)
    print(solv_dir)

    if os.path.exists(solv_dir) == False:
        os.mkdir(solv_dir)
    if restart == True:
        pass
    else:
        try:
            if os.path.exists(vac_dir + '/vasprun.xml'):
                print('FOUND a VACUUM RUN')
                vac_vasprun = read_vasprun(vac_dir)
                print(vac_vasprun.converged)
                if vac_vasprun.converged == True:
                    copy_vacuum_files(directory, solv_dir, restart=restart)
                else:
                    print('vasprun converged == False')
                    copy_directory_files(directory, solvent=solvent)
            else:
                copy_directory_files(directory, solvent=solvent)
        except:
            print('FOUND ERROR IN VAC VASPRUN, COPYING FILES ANYWAY')
            copy_vacuum_files(directory, solv_dir, restart=restart)

    if os.path.exists(solv_dir + '/CONTCAR'):
        sh.copy(solv_dir + '/CONTCAR', solv_dir + '/POSCAR')
    os.chdir(solv_dir)
    #add_gdc2(solv_dir, dz=dz, epsilon=epsilon, Ionic_strength=Ionic_strength, NSW=NSW, dict_of_additional_tags = dict_of_additional_tags)
    #add_solvent(solv_dir, solvent=solvent, calc_type=calc_type, NSW=NSW)
    add_solvent(directory, solvent=solvent, Ionic_strength = Ionic_strength, calc_type=calc_type, NSW=NSW, dict_of_additional_tags=dict_of_additional_tags)
    os.chdir(cwd)
    return solv_dir

def vacuum_init_solvent(directory, 
                        solv_dir, 
                        solvent='water', 
                        epsilon=78.4,
                        Ionic_strength = 1, 
                        calc_type='linear', 
                        NSW=100, 
                        dict_of_additional_tags={}):
    directory = os.path.abspath(directory)
    solv_dir = os.path.abspath(solv_dir)
    initialize_solvent2(directory, solvent=solvent,calc_type=calc_type, epsilon=epsilon, Ionic_strength=Ionic_strength, NSW=NSW, dict_of_additional_tags=dict_of_additional_tags)
    remove_Vaspsol_tags(solv_dir)
    #run(solv_dir) # CHANGED ON 8/31/2022 TO SPEED UP, COULD CAUSE CONVERGENCE ISSUES 
    sh.copy(solv_dir + '/CONTCAR', solv_dir +'/POSCAR')
    add_solvent(solv_dir, solvent=solvent, Ionic_strength = Ionic_strength, calc_type=calc_type, NSW=NSW, dict_of_additional_tags=dict_of_additional_tags)
    return 


def calculate_solvent(directory, solvent='water', calc_type='linear', epsilon=78.4, Ionic_strength=0, restart=False, NSW=100, dict_of_additional_tags={}):
    cwd = os.getcwd()
    directory = os.path.abspath(directory)
    if solvent != 'water':
        try:
            props = solvent_properties(solvent)
            epsilon = props['EB_K']
        except:
            pass
    solv_dir = directory + '/Solvent_' + str(solvent) + '_epsilson_' + str(epsilon) + '_Ionic_' + str(Ionic_strength) + '_calc_' + str(calc_type)
    tags = list(dict_of_additional_tags.keys())
    for tag in tags:
        solv_dir = solv_dir+'_'+tag+'_'+str(dict_of_additional_tags[tag])
    solv_dir = os.path.abspath(solv_dir)

    os.chdir(directory)
    if os.path.exists(solv_dir + '/CONTCAR')==True or restart == True:
        sh.copy(solv_dir + '/CONTCAR', solv_dir + '/POSCAR')
        add_solvent(solv_dir, solvent=solvent, Ionic_strength = Ionic_strength, calc_type=calc_type, NSW=NSW, dict_of_additional_tags=dict_of_additional_tags)
        print('PREVIOUSLY RUN JOB. RESTARTING')
    else:
        # if the solvent directory exists: 1. check if it is converged 2. check if files can be imported from VAC 3. if no import from directory
        vacuum_init_solvent(directory,
                        solv_dir, 
                        solvent=solvent, 
                        epsilon = epsilon,
                        Ionic_strength = Ionic_strength, 
                        calc_type=calc_type, 
                        NSW=NSW, 
                        dict_of_additional_tags=dict_of_additional_tags)
        
        #initialize_solvent2(directory, calc_type=calc_type, epsilon=epsilon, Ionic_strength=Ionic_strength, NSW=NSW, dict_of_additional_tags=dict_of_additional_tags)
    print('STARTING VASP SOLVATION CALC')
    if calc_type=='non-linear':
        run_nln(solv_dir)
    else:
        run(solv_dir)
    try:
        vasprun = read_vasprun(solv_dir)
        os.chdir(cwd)
        return vasprun
    except:
        os.chdir(cwd)
    return

def calculate_gdc(directory, dz=5, epsilon=78.4, Ionic_strength=1, restart=False, NSW=100, dict_of_additional_tags={}):
    cwd = os.getcwd()
    directory = os.path.abspath(directory)
    solvent = 'dz_' + str(dz) + '_epsilson_' + str(epsilon) + '_Ionic_' + str(Ionic_strength)
    tags = list(dict_of_additional_tags.keys())
    for tag in tags:
        solvent = solvent+'_'+tag+'_'+str(dict_of_additional_tags[tag])
    solv_dir = directory + '/' + solvent
    solv_dir = os.path.abspath(solv_dir)
    os.chdir(directory)

    keep_files = ['POSCAR','POTCAR','WAVECAR','CHGCAR','INCAR']
    if restart == True:
        try:
            os.remove(solv_dir + '/vasprun.xml')
        except:
            pass
    # if the solvent directory exists: 1. check if it is converged 2. check if files can be imported from VAC 3. if no import from directory
    initialize_gdc(directory, dz=dz, epsilon=epsilon, Ionic_strength=Ionic_strength, NSW=NSW, restart=restart, dict_of_additional_tags=dict_of_additional_tags)
    remove_Vaspsol_tags(solv_dir)
    run(solv_dir)
    sh.copy(solv_dir + '/CONTCAR', solv_dir +'/POSCAR')
    add_gdc2(solv_dir, dz=dz, epsilon=epsilon, Ionic_strength=Ionic_strength, NSW=NSW, dict_of_additional_tags = dict_of_additional_tags)
    run_gdc(solv_dir)
    os.chdir(cwd)
    try:
        vasprun = read_vasprun(solv_dir)
        return vasprun
    except:
        return

def calculate_solvation_energy(directory, solvent='water', calc_type='linear', epsilon=78.4, Ionic_strength=0, restart=False, NSW=100, dict_of_additional_tags={}):
    cwd = os.getcwd()
    directory = os.path.abspath(directory)
    vacuum_run = calculate_vacuum(directory, restart=restart, NSW=NSW, dict_of_additional_tags=dict_of_additional_tags)
    solvent_run = calculate_solvent(directory, solvent=solvent, calc_type=calc_type, epsilon=epsilon, Ionic_strength=Ionic_strength, restart=restart, NSW=NSW, dict_of_additional_tags={})
    solvation_energy = solvent_run.final_energy - vacuum_run.final_energy
    os.chdir(directory)
    os.system('rm */CHG*')
    os.system('rm */WAVECAR')
    os.chdir(cwd)
    return solvation_energy, vacuum_run, solvent_run

def get_solvation_energy(directory, solvent='water'):
    cwd = os.getcwd()
    directory = os.path.abspath(directory)
    os.chdir(directory)
    vac = Vasprun('./VAC/vasprun.xml')
    oszicar_vac = Oszicar('./VAC/OSZICAR').final_energy
    solv_dirs = [i for i in next(os.walk(directory))[1] if solvent in i]
    solv_energy = []
    oszicar_solv_energies = []
    for solv_dir in solv_dirs:
        try:
            #solv_run = read_vasprun(solv_dir)
            #solv_energy.append(solv_run.final_energy - vac.final_energy)
            oszicar_solv_energies.append(Oszicar(solv_dir + '/OSZICAR').final_energy - oszicar_vac)
            solv_energy.append(Oszicar(solv_dir + '/OSZICAR').final_energy - oszicar_vac)
            #if np.abs(oszicar_solv_energies[-1]) < np.abs(solv_energy[-1]):
            #    solv_energy[-1] = oszicar_solv_energies[-1]
        except:
            pass
    return solv_energy

def get_mu_gdc():
    struc = Structure.from_file('POSCAR')
    positions = struc.cart_coords
    z_coord = positions[::,2]
    distances = []
    for i in z_coord:
        for j in z_coord:
            distances.append(i-j)
    distances = np.array(distances).reshape(len(z_coord), len(z_coord))
    max_dist = np.max(distances)
    index = np.where(distances==max_dist)[0][0]
    print(index)
    print(z_coord)
    c = struc.lattice.c
    mu = (c - max_dist)/2 + z_coord[index]
    return mu

def add_gdc(dz):
    a, b = get_bounds_gdc(dz)
    ef.Replace_Incar('S_Z_START',a)
    ef.Replace_Incar('S_Z_END',b)
    return

def add_gdc2(directory, dz=5, epsilon=78.4, Ionic_strength=1, NSW=100, dict_of_additional_tags={}):
    cwd = os.getcwd()
    os.chdir(directory)
    Ionic_strength = 3.04 / Ionic_strength**(1/2)
    ef.Replace_Incar('EB_K',epsilon)
    ef.Replace_Incar('LAMBDA_D_K',Ionic_strength)
    ef.Replace_Incar('NSW',NSW)
    a, b = get_bounds_gdc(dz)
    ef.Replace_Incar('S_Z_START',a)
    ef.Replace_Incar('S_Z_END',b)
    ef.Replace_Incar('GDC_SIGMA',2)
    ef.Replace_Incar('LSOL', 'TRUE')
    ef.Replace_Incar('TAU', '0.000')
    ef.Replace_Incar('LRHOION', 'TRUE')
    ef.Replace_Incar('LRHOB', 'TRUE')
    dict_keys = list(dict_of_additional_tags.keys())
    for key in dict_keys:
        ef.Replace_Incar(key, dict_of_additional_tags[key])
    os.chdir(cwd)
    return

def get_bounds_gdc(dz):
    struc = Structure.from_file('POSCAR')
    mu = get_mu_gdc()
    a = mu-dz/2
    b = mu+dz/2
    c = struc.lattice.c
    if a < 0:
        a = a+c
    if b > c:
        b = b-c
    return a, b

def run(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    os.system('module load intel openmpi')
    command = 'mpirun --allow-run-as-root /blue/hennig/ericfonseca/VASPsol_compilations/VASP6/bin/vasp_std'
    #command = 'sudo mpirun --allow-run-as-root ~/vasp/bin/vasp_std'
    os.system(command)
    os.chdir(cwd)
    return 


def run_gdc(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    os.system('module load intel openmpi')
    command = 'mpirun --allow-run-as-root /blue/hennig/ericfonseca/VASPsol_compilations/GDCsol/bin/vasp_std'
    os.system(command)
    #p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
   # p.wait()
    #p.stdout.close()
    os.chdir(cwd)
    return

def run_nln(directory):
    cwd = os.getcwd()
    os.chdir(directory)
    ef.Replace_Incar('LNLSOL','TRUE')
    os.system('module load intel openmpi')
    command = 'mpirun --allow-run-as-root /blue/hennig/ericfonseca/NASA/VASPsol/angel/NLN_5.4.4/bin/vasp_std'
    os.system(command)
    os.chdir(cwd)
    return

def replace_vasprun(directory):
    directory = os.path.abspath(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        with open('./vasprun.xml') as vasprun:
            vasprun_data = vasprun.read()
        vasprun_data = vasprun_data.replace('<i name="LAMBDA_D_K">****************</i>','<i name="LAMBDA_D_K">     0.00000</i>')
        with open('./vasprun.xml','w') as vasprun:
            vasprun.write(vasprun_data)
        print('FIXED LAMBDA_D_K correct in file')
    except:
        print('LAMBDA_D_K correct in file')
    os.chdir(cwd)
    return

def read_vasprun(directory):
    directory = os.path.abspath(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    replace_vasprun(directory)
    vasprun = Vasprun('./vasprun.xml')
    os.chdir(cwd)
    return vasprun

def calculate_surface_energy(bulk_directory, slab_directory):
    bulk_directory = os.path.abspath(bulk_directory)
    slab_directory = os.path.abspath(slab_directory)
    bulk_vasprun = read_vasprun(bulk_directory)
    slab_vasprun = read_vasprun(slab_directory)
    slab_struc = slab_vasprun.structures[-1]
    natoms_bulk = len(bulk_vasprun.structures[-1].sites)
    natoms_slab = len(slab_struc.sites)
    bulk_energy = bulk_vasprun.final_energy
    slab_energy = slab_vasprun.final_energy
    slab_area = slab_struc.lattice.a * slab_struc.lattice.b * np.sin(slab_struc.lattice.gamma * np.pi / 180)
    surface_energy = (slab_energy - natoms_slab*(bulk_energy / natoms_bulk))/2/slab_area
    return surface_energy, bulk_vasprun, slab_vasprun

def run_solvents(directory, solvents):
    jobs = []
    directory = os.path.abspath(directory)
    vac = calculate_vacuum(directory)
    for i in solvents:
        jobs.append(dask.delayed(calculate_solvent)(directory, solvent=i))
    jobs = dask.compute(jobs)
    return vac, jobs


def get_truhler_dataframe():
    csv = pd.read_excel('/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/MNSol_alldata.xls')
    return csv

def run_molecule(directory):
    cwd = os.getcwd()
    directory = os.path.abspath(directory)
    os.chdir(directory)
    filehandle = directory.split('/')[-1]
    csv = get_truhler_dataframe()
    solvents = csv[csv['FileHandle'] == filehandle]['Solvent'].to_numpy().astype(np.str_)
    jobs = []
    try:
        jobs = run_solvents(directory, solvents)
    except:
        print(directory, ' FAILED')
        pass
    os.chdir(cwd)
    return jobs

def copy_INCAR_KPOINTS():
    files = [os.path.abspath(i) for i in next(os.walk('.'))[1]]
    for i in files:
        sh.copy('INCAR', i+'/INCAR')
        sh.copy('KPOINTS',i+'/KPOINTS')
        sub_files = [os.path.abspath(i+'/'+file) for file in next(os.walk(i))[1]]
        for j in sub_files:
            print(j)
            sh.copy('INCAR', j+'/INCAR')
            sh.copy('KPOINTS', j+'/KPOINTS')


def get_energies(directory):
    #solvents = df[df['FileHandle'] == filehandle].to_numpy()
    directory = os.path.abspath(directory)
    filehandle = directory.split('/')[-1]
    sub_dirs = next(os.walk(filehandle))[1]
    print(sub_dirs)
    solvents = [i.split('_')[1] for i in sub_dirs]
    print(solvents)
    energies = []
    for solvent in solvents:
        energies.append(get_solvation_energy(filehandle, solvent=solvent))
    return solvents, energies

def get_final_eng(pathToOszicar):
    directory = os.path.abspath(pathToOszicar)
    try:
        with open(directory+'/OSZICAR','r') as o:
            lines = o.readlines()
        F = [line for line in lines if 'F=' in line]
        eng = float(F[-1].split()[2])
    except:
        eng = np.nan
    return eng



class data():
    def __init__(self, directory):
        directory = os.path.abspath(directory)
        files = [os.path.abspath(directory+'/'+i) for i in next(os.walk(directory))[1] if 'Solvent' in i]
        self.files = files
        self.filehandles = np.array([directory.split('/')[-1] for i in files])
        self.solvent = np.array([i.split('Solvent_')[-1].split('_')[0] for i in files])
        self.files = files
        nan_array = np.array([np.nan for file in files])
        self.directory = directory
        self.nc_k = np.array([0.0025 for file in files])
        self.sigma_k = np.array([0.6 for file in files])
        self.energies = np.array([np.nan for file in files])
        self.tau = np.array([0.000525 for file in files])
        df = pd.DataFrame()
        df['file'] = files
        df['Solvent'] = self.solvent
        df['FileHandle'] = self.filehandles
        df['NC_K'] = self.nc_k
        df['SIGMA_K'] = self.sigma_k
        df['TAU'] = self.tau
        df['Solvation_Energy'] = nan_array
        df['Total_Energy'] = self.energies
        self.df = df

        # Past init of dataclass
        for i, file in enumerate(files):
            print(file)
            try:
                self.tau[i] = float(file.split('TAU_')[-1].split('_')[0].split('/')[0])
            except:
                pass
        for i, file in enumerate(files):
            try:
                self.nc_k[i] = float(file.split('NC_K_')[-1].split('_')[0].split('/')[0])
            except:
                pass
        for i, file in enumerate(files):
            try:
                self.sigma_k[i] = float(file.split('SIGMA_K_')[-1].split('_')[0]) 
            except:
                pass
        for i, file in enumerate(files):
            try:
                self.energies[i] = get_final_eng(file)
            except:
                pass
        self.energies = np.array(self.energies)
        self.vac_energy = get_final_eng(directory+'/VAC')
        try:
            self.solv_energies = self.energies-self.vac_energy
        except:
            print('COUNT NOT LOAD ','SOLVATION_ENERGIES')
            pass
        csv = get_truhler_dataframe()
        df = pd.DataFrame()
        df['Solvent'] = self.solvent
        df['FileHandle'] = self.filehandles
        df['NC_K'] = self.nc_k
        df['SIGMA_K'] = self.sigma_k
        df['TAU'] = self.tau
        df['Solvation_Energy'] = self.solv_energies
        df['Total_Energy'] = self.energies
        self.df = df
        self.ml_df = pd.merge(df, csv, on=['Solvent', 'FileHandle'])
        self.ml_df['error'] = self.ml_df['DeltaGsolv'] - self.ml_df['Solvation_Energy']*23.06
        self.ml_df['error_ev'] = self.ml_df['DeltaGsolv']/23.06 - self.ml_df['Solvation_Energy']
        self.ml_df['error_frac'] = self.ml_df['error'] / self.ml_df['DeltaGsolv']
        return
 
def get_mean_df(df):
    df2 = df.groupby(['NC_K','SIGMA_K','TAU']).mean()['error']
    nc_k = [i[0] for i in df2.index.values]
    sigma_k = [i[1] for i in df2.index.values]
    tau = [i[2] for i in df2.index.values]
    df = pd.DataFrame()
    df['NC_K'] = nc_k
    df['SIGMA_K'] = sigma_k
    df['TAU'] = tau
    df['error'] = df2.to_numpy().flatten()
    return df


def plot_cf(df):
    try:
        title = df['SoluteName'].unique()[0]
    except:
        title = 'NA'
    #df = get_mean_df(df)
    x = df['NC_K'].to_numpy().reshape(-1,1)
    y = df['SIGMA_K'].to_numpy().reshape(-1,1)
    z = np.abs(df['error'].to_numpy()).reshape(-1,1)
    index = np.where(z==z.min())
    df2 = pd.DataFrame()
    df2['NC_K'] = x.reshape(-1)
    df2['SIGMA_K'] = y.reshape(-1)
    df2['error'] = z.reshape(-1)
    x1 = np.linspace(df2['NC_K'].min(), df2['NC_K'].max(),1000)#len(df2['NC_K'].unique()))
    y1 = np.linspace(df2['SIGMA_K'].min(), df2['SIGMA_K'].max(),1000)# len(df2['SIGMA_K'].unique()))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata(np.hstack([x,y]), z, (x2, y2))
    z2 = z2.reshape(x2.shape)
    fig = plt.subplots(dpi=300)
    cs = plt.contourf(x2, y2, z2)#, locator=ticker.LogLocator())
    plt.scatter(x, y, s=10, c='k')
    plt.scatter(x[index], y[index], s=10, c='r')
    print('MIN ERROR: ',z[index])
    plt.xlabel('NC_K $\dfrac{e}{Ang^3}$')
    plt.ylabel('SIGMA_K')
    #plt.xlim([1e-3, 1e-2])
    plt.colorbar(cs)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(title+'_Grid_Search.png')
    plt.show()
    return plt.gca()



def plot_3d(df):
    x = df['NC_K'].to_numpy().reshape(-1,1)
    y = df['SIGMA_K'].to_numpy().reshape(-1,1)
    z = np.abs(df['error'].to_numpy()).reshape(-1,1)
    df2 = pd.DataFrame()
    df2['NC_K'] = x.reshape(-1)
    df2['SIGMA_K'] = y.reshape(-1)
    df2['error'] = z.reshape(-1)
    x1 = np.linspace(df2['NC_K'].min(), df2['NC_K'].max(), len(df2['NC_K'].unique()))
    y1 = np.linspace(df2['SIGMA_K'].min(), df2['SIGMA_K'].max(), len(df2['SIGMA_K'].unique()))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata(np.hstack([x,y]), z, (x2, y2))
    z2 = z2.reshape(x2.shape)
    fig = plt.subplot()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1)
    plt.xlabel('NC_K $\dfrac{e}{Ang^3}$')
    plt.ylabel('SIGMA_K')
    ax.set_zlabel('Energy Deviation')
    plt.savefig('Grid_Search.png')
    plt.show()


def xyz_to_POSCAR(fname):
    temp = io.read(fname)
    temp = ef.ase_to_pymatgen(temp)
    s = Structure([[100,0,0],[0,100,0],[0,0,100]], 
                        temp.species, 
                        temp.cart_coords,
                        coords_are_cartesian = True)
    s.to('POSCAR', fname.split('.')[0] + '_POSCAR')
    return s
