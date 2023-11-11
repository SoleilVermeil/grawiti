import random
import kwant
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import math
from tqdm import tqdm
import shutil
import tarfile

def make_system(
    scatter_width: int,
    scatter_length: int,
    lead_width: int,
    lead_length: int = 1,
    scatter_up_shift: int = 0,
    left_lead_up_shift: int = 0,
    right_lead_up_shift: int = 0,
    verbose: bool = False,
) -> tuple:
    """
    Creates a kwant system, made of a simple rectangular scattering region and two leads, one on each side.
    
    Parameters
    ----------
    * scatter_width (int): width of the scattering region, in layers.
    * scatter_length (int): length of the scattering region, in number of 'honeycomb' cells (ie. 3N layers).
    * lead_width (int): width of the leads, in layers.
    * lead_length (int): length of the leads, in number of 'honeycomb' cells (ie. 3N layers).
    * scatter_up_shift (int): number of layers to shift the scattering region up by.
    * left_lead_up_shift (int): number of layers to shift the left lead up by.
    * right_lead_up_shift (int): number of layers to shift the right lead up by.
    * verbose (bool): whether to print the system parameters or not.
    
    Returns
    -------
    * sys (kwant.builder.Builder): the scattering region.
    * left_lead (kwant.builder.Builder): the left lead.
    * right_lead (kwant.builder.Builder): the right lead.
    """

    if verbose: print("Building a system with the following parameters:")
    if verbose: print(f"Scatter region: {scatter_width:>2} layers wide & {scatter_length:>2} 'honeycomb' long (ie. {4*scatter_length:>2} layers long)")
    if verbose: print(f"Lead region: {lead_width:>2} layers wide & {lead_length:>2} 'honeycomb' long (ie. {4*lead_length:>2} layers long)")

    # Defining constants
    # ------------------

    SIN_60 = math.sin(math.pi / 3)
    GRAPHENE = kwant.lattice.general(
        [(0, math.sqrt(3)), (-3/2, -math.sqrt(3)/2)], # base vectors
        [(0,0), (1/2,math.sqrt(3)/2)], # initial atoms (in cartesian coordinates), one for each atom in the unit cell
        norbs = 1, # number of orbitals per site
    )
    POT = 0.0
    T = 1
    SCATTER_LENGTH = 3 * scatter_length
    LEAD_LENGTH = 3 * lead_length
    SMALL = 1e-3

    # Defining the scattering region
    # ------------------------------

    def scatter_rect(pos):
        x, y = pos
        # y -= scatter_up_shift * SIN_60
        # is_inside_rectangle = ((0 < x + SMALL < SCATTER_LENGTH) and (0 < y + SMALL < scatter_width * SIN_60))
        is_inside_rectangle = ((0 < x + SMALL < SCATTER_LENGTH) and (scatter_up_shift * SIN_60 < y + SMALL < scatter_width * SIN_60 + scatter_up_shift * SIN_60))

        return is_inside_rectangle
    sys = kwant.Builder()
    sys[GRAPHENE.shape(scatter_rect, start=(0 if scatter_up_shift % 2 == 0 else 1/2, 0 * scatter_up_shift * SIN_60))] = POT
    sys[GRAPHENE.neighbors()] = -T

    # Defining the left
    # -----------------

    if verbose: print("Lead will be inside [{};{}]x[{};{}]".format(0, LEAD_LENGTH, left_lead_up_shift * SIN_60, lead_width * SIN_60 + left_lead_up_shift * SIN_60))
    def lead_rect(pos):
        x, y = pos
        # y -= left_lead_up_shift * SIN_60
        # is_inside_rectangle = ((0 < x + SMALL < LEAD_LENGTH) and (0 < y + SMALL < lead_width * SIN_60))
        is_inside_rectangle = ((0 < x + SMALL < LEAD_LENGTH) and (left_lead_up_shift * SIN_60 < y + SMALL < lead_width * SIN_60 + left_lead_up_shift * SIN_60))
        return is_inside_rectangle
    sym = kwant.TranslationalSymmetry((LEAD_LENGTH, 0))
    left_lead = kwant.Builder(sym)
    left_lead[GRAPHENE.shape(lead_rect, start=(0 if left_lead_up_shift % 2 == 0 else 1/2, left_lead_up_shift * SIN_60))] = POT
    left_lead[GRAPHENE.neighbors()] = -T

    # Defining the right lead
    # -----------------------

    def lead_rect_right(pos):
        x, y = pos
        # y -= right_lead_up_shift * SIN_60
        # is_inside_rectangle = ((0 < x + SMALL < LEAD_LENGTH) and (0 < y + SMALL < lead_width * SIN_60))
        is_inside_rectangle = ((0 < x + SMALL < LEAD_LENGTH) and (right_lead_up_shift * SIN_60 < y + SMALL < lead_width * SIN_60 + right_lead_up_shift * SIN_60))
        return is_inside_rectangle
    sym = kwant.TranslationalSymmetry((LEAD_LENGTH, 0))
    right_lead = kwant.Builder(sym)
    right_lead[GRAPHENE.shape(lead_rect_right, start=(0 if right_lead_up_shift % 2 == 0 else 1/2, right_lead_up_shift * SIN_60))] = POT
    right_lead[GRAPHENE.neighbors()] = -T

    return sys, left_lead, right_lead

def compute_conductance(fsyst, energies: list, t: float) -> list:
    conductance = []
    SMALL = 1e-10
    for energy in energies:
        smatrix = kwant.smatrix(fsyst, energy * t + SMALL)
        conductance.append(smatrix.transmission(1, 0))
    return conductance


def generate_junction():
    
    # Defining the system
    # -------------------
    
    scatter_width = random.randint(3, 10)
    scatter_length = random.randint(1, 5) + 0.5 * random.randint(0, 1)
    lead_width = 2 # NOTE: Other values are not supported yet
    scatter_up_shift = random.randint(-1, 1)
    left_lead_up_shift = random.randint(0, math.floor(scatter_width / 2) * 2 - 1)
    right_lead_up_shift = random.randint(0, math.floor(scatter_width / 2) * 2 - 1)
    try:
        logging.info(f"Building a system with the following parameters:")
        logging.info(f"* scatter_width =       {scatter_width}")
        logging.info(f"* scatter_length =      {scatter_length}")
        logging.info(f"* lead_width =          {lead_width}")
        logging.info(f"* scatter_up_shift =    {scatter_up_shift}")
        logging.info(f"* left_lead_up_shift =  {left_lead_up_shift}")
        logging.info(f"* right_lead_up_shift = {right_lead_up_shift}")
        syst, left_lead, right_lead = make_system(
            scatter_width = scatter_width, # For a simple molecule: 3
            scatter_length = scatter_length, # For a simple molecule: 1
            lead_width = lead_width, # For a simple molecule: 2
            scatter_up_shift = scatter_up_shift, # For a simple molecule: 1
            left_lead_up_shift = left_lead_up_shift, # For a simple molecule: see table above
            right_lead_up_shift = right_lead_up_shift, # For a simple molecule: see table above
        )
        syst.attach_lead(left_lead.reversed())
        syst.attach_lead(right_lead)
        fsyst = syst.finalized()
        t = 2.75
        conductance = compute_conductance(fsyst, energies=[0], t=t)
    
        if False:
            _, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)
            kwant.plot(syst, ax=ax)
            # set equal aspect
            ax.set_aspect("equal")
            # ax.set_xlim(-2, 5)
            # ax.set_ylim(-5, 5)
            # plt.savefig(f"data/junction.png")
    except ValueError as e:
        logging.warning("Could not make the system. Skipping.")
        return False
        
    # Building adjacancy matrix
    # -------------------------
    
    hamiltonian = fsyst.hamiltonian_submatrix(sparse=False)
    adjacancy_matrix = 1 * (hamiltonian != 0)
    
    # Save adjacancy matrix
    # ---------------------
    
    name = f"junction_{random.randint(10000000, 99999999)}"
    if not os.path.exists(f"data/{name}"):
        os.makedirs(f"data/{name}")
    np.savetxt(f"data/{name}/adjacancy.txt", adjacancy_matrix, fmt="%d")
    
    # Save junction properties
    # ---------------------------
    
    lead_interfaces = fsyst.lead_interfaces
    interface_indices = [i[0] for i in lead_interfaces]
    i = interface_indices[0]
    j = interface_indices[1]
    pos = np.array([s.pos for s in fsyst.sites])
    
    properties = {
        "name": name,
        "conductance": float(conductance[0]),
        "i": int(i),
        "j": int(j),
        "sites": [ 
            {
                "index": int(i),
                "x": float(pos[i][0]),
                "y": float(pos[i][1]),
            } for i in range(len(pos))
        ],
    }        
    json.dump(obj=properties, fp=open(f"data/{name}/properties.json", "w"), indent=4)
    

if __name__ == '__main__':
    
    if os.path.exists("data"):
        answer = input("Do you want to delete all previous data? [y/n] ")
        if answer == "y":
            shutil.rmtree("data")
            os.makedirs("data")
        elif answer == "n":
            pass
        else:
            raise ValueError("Invalid answer.")
    
    for _ in tqdm(range(1000)):
        generate_junction()
        
    with tarfile.open("data.tar.gz", "w:gz") as tar:
        tar.add("data")
        
    shutil.rmtree("data")