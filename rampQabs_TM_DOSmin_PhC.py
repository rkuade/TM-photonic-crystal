import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time,sys,argparse

from objective_TM_DOS_PhC import designdof_ldos_objective
import ceviche
from ceviche.constants import C_0, ETA_0

import time,sys,argparse

import nlopt


parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-pow10Qabs_start',action='store',type=float,default=2.0)
parser.add_argument('-pow10Qabs_end',action='store',type=float,default=6.0)
parser.add_argument('-pow10Qabs_num',action='store',type=int,default=5)

parser.add_argument('-ReChi',action='store',type=float,default=2.0)
parser.add_argument('-ImChi',action='store',type=float,default=1e-2)
parser.add_argument('-gpr',action='store',type=int,default=20)

###design area size, design area is rectangular with central rectangular hole where the dipole could live; for a photonic crystal, the entire design region is filled with dipoles ###
parser.add_argument('-design_x',action='store',type=float,default=1.0)
parser.add_argument('-design_y',action='store',type=float,default=1.0)
parser.add_argument('-Num_Poles', action='store', type=int, default=1)
parser.add_argument('-geometry', action='store', type=str, default='Cavity')

parser.add_argument('-vacuum_x',action='store',type=float,default=0.2)
parser.add_argument('-vacuum_y',action='store',type=float,default=0.2)

parser.add_argument('-emitter_x',action='store',type=float,default=0.05)
parser.add_argument('-emitter_y',action='store',type=float,default=0.05)

parser.add_argument('-dist_x',action='store',type=float,default=0.02)

#separation between pml inner boundary and source walls
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-init_type',action='store',type=str,default='vac')
parser.add_argument('-init_file',action='store',type=str,default='test.txt')
parser.add_argument('-maxeval',action='store',type=int,default=10000)
parser.add_argument('-output_base',action='store',type=int,default=10)
parser.add_argument('-name',action='store',type=str,default='test')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

opt_data = {'count':0, 'output_base':args.output_base, 'name':args.name}

# construct base frequency omega_0 and grid increments; 0.4 is for agreement
# with an example from the book by John Joannopoulos called: "Photonic Crystals
# Molding the Flow of Light"  
k = 2*np.pi/args.wavelength
omega = 0.4 * C_0 * k
dl = 1.0/args.gpr

# construct design region
design_vals = [args.design_x]
for design_i in range(len(design_vals)):
    Mx = int(np.round(design_vals[design_i]/dl))
    My = int(np.round(design_vals[design_i]/dl))
    Npml = int(np.round(args.pml_thick/dl))
    Npmlsep = int(np.round(args.pml_sep/dl))
    Emitterx = int(np.round(args.emitter_x / dl))
    Emittery = int(np.round(args.emitter_y / dl))
    Vacuumx = int(np.round(args.vacuum_x / dl))
    Vacuumy = int(np.round(args.vacuum_y / dl))
    Distx = int(np.round(args.dist_x / dl))


    Nx = Mx + 2*(Npmlsep+Npml)+Distx
    Ny = My + 2*(Npmlsep+Npml)

    # define design region
    design_mask = np.zeros((Nx,Ny), dtype=bool)
    design_mask[Npml+Npmlsep+Distx:Npml+Npmlsep+Distx+Mx , Npml+Npmlsep:Npml+Npmlsep+My] = True
    design_mask[Npml+Npmlsep+Distx+(Mx-Vacuumx)//2:Npml+Npmlsep+Distx+(Mx-Vacuumx)//2+Vacuumx,Npml+Npmlsep+(My-Vacuumy)//2:Npml+Npmlsep+(My-Vacuumy)//2+Vacuumy] = False
   
    # define susceptibility and relative permittivity and emitter (takes up
    # entire design region for a photonic crystal)
    chi = args.ReChi - 1j*args.ImChi #ceviche has +iwt time convention
    epsval = 1.0 + chi
    print('epsval', epsval, flush=True)
    if args.geometry.lower()[0] == 'c':
        print('cavity optimization', flush=True)
        emitter_mask = np.zeros((Nx,Ny), dtype=bool)
        emitter_mask[Npml+Npmlsep+(Mx-Emitterx)//2:Npml+Npmlsep+(Mx-Emitterx)//2+Emitterx,Npml+Npmlsep+(My-Emittery)//2:Npml+Npmlsep+(My-Emittery)//2+Emittery] = True
    else:
        print('half-space optimization', flush=True)
        emitter_mask = np.zeros((Nx,Ny), dtype=bool)
        emitter_mask[Npml+Npmlsep:Npml+Npmlsep+Emitterx,Npml+Npmlsep+(My-Emittery)//2:Npml+Npmlsep+(My-Emittery)//2+Emittery] = True

    #set TM dipole source
    source = np.zeros((Nx,Ny), dtype=complex)
    source[emitter_mask] = 1.0 / (dl*dl)

    # solve for the electric field in the z direction
    epsVac = np.ones((Nx,Ny), dtype=complex)
    sim_vac = ceviche.fdfd_ez(omega, dl, epsVac, [Npml,Npml])
    _,_,vac_field = sim_vac.solve(source)

    # calculate LDOS from fields that were obtained at the central frequency
    vac_ldos = np.real(np.sum(np.conj(source)*vac_field)) * 0.5 * dl**2
    opt_data['vac_ldos'] = vac_ldos
    print('vacuum LDOS', vac_ldos)
    

    #check configuration
    config = np.zeros((Nx,Ny))
    config[design_mask] = 1.0
    config[emitter_mask] = 2.0
    plt.imshow(config)
    plt.savefig(args.name+str(design_vals[design_i])+'_check_config.png')

    
    # initialize design region with material
    ndof = np.sum(design_mask)
    if args.init_type=='vac':
        designdof = np.zeros(ndof)
    if args.init_type=='slab':
        designdof = np.ones(ndof)
    if args.init_type=='half':
        designdof = 0.5*np.ones(ndof)
    if args.init_type=='rand':
        designdof = np.random.rand(ndof)
    if args.init_type=='file':
        designdof = np.loadtxt(args.init_file)

    Qabslist = 10.0**np.linspace(args.pow10Qabs_start, args.pow10Qabs_end, args.pow10Qabs_num)

    # run optimization for various quality factors (quality factors represent
    # domega - the width of the frequency window over which to establish a 
    # photonic bandgap)
    for Qabs in Qabslist:
        print('at Qabs', Qabs)
        opt_data['count'] = 0 #refresh the iteration count
    
        if Qabs<1e16:
            omega_Qabs = omega * (1-1j/2/Qabs)
            opt_data['name'] = args.name + f'_Qabs{Qabs:.1e}' + 'Ly'+str()
        else:
            omega_Qabs = omega
            opt_data['name'] = args.name + '_Qinf'
        
        # set up objective function 
        optfunc = lambda dof, grad: designdof_ldos_objective(dof, grad, epsval, design_mask, dl, source, omega, args.Num_Poles, Qabs, epsVac, Npml, opt_data)

        # define and set up lower and upper bounds for the degrees of freedom, 
        # which are mapped onto the relative permittivity at each grid point 
        # (lower bound transforms from 0 to the relative permittivity of vacuum 
        # and upper bound transforms from 1 to the relative permittivity of the 
        # material that we set with chi; use the method of moving asymptotes
        # for the optimization
        lb = np.zeros(ndof)
        ub = np.ones(ndof)
 
        opt = nlopt.opt(nlopt.LD_MMA, int(ndof))
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        if Qabs < 1e4:
            opt.set_xtol_rel(1e-8)
        else:
            opt.set_xtol_rel(1e-11)
        opt.set_maxeval(args.maxeval)

        opt.set_min_objective(optfunc)
        omegas = []
        sim_vacs = []
        vac_ldos = 0
        Polefactor = 0
        # compute the reference vacuum ldos with the same 
        # number of poles as used for the optimization
        for nn in range(args.Num_Poles):
            Polefactor += -1j*np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*args.Num_Poles))
        for nn in range(args.Num_Poles):
            omegas += [omega * (1-np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*args.Num_Poles))/2./Qabs)]
            sim_vacs += [ceviche.fdfd_ez(omegas[nn], dl, epsVac, [Npml,Npml])]
            _,_,vac_field = sim_vacs[nn].solve(source)
            vac_ldos += dl**2 * 0.5 * np.real(-1j*(np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*args.Num_Poles))/Polefactor*np.sum(np.conj(source) * vac_field)))
            
        designdof = opt.optimize(designdof.flatten())
        min_ldos = opt.last_optimum_value()
        min_enh = min_ldos / vac_ldos
        print('vacuum LDOS', vac_ldos)

        print(f'Qabs{Qabs:.1e} best LDOS and enhancement found via topology optimization', min_ldos, min_enh)
        
        # plot and save optimal design
        np.savetxt(opt_data['name'] +str(design_vals[design_i])+ '_optdof.txt', designdof)


        opt_design = np.zeros((Nx,Ny))
        opt_design[design_mask] = designdof
        plt.figure()
        plt.imshow(np.reshape(opt_design[Npml+Npmlsep+Distx:Npml+Npmlsep+Distx+Mx,Npml+Npmlsep:Npml+Npmlsep+My], (Mx,My)))
        plt.savefig(opt_data['name']+str(design_vals[design_i])+'_opt_design.png')
    

