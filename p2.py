import numpy as np
import matplotlib.pyplot as plt
import utils as utl
import os
from exoctk.limb_darkening.spam import transform_coefficients as tcs
import exoctk
from tqdm import tqdm

p1 = os.getcwd()
p1r = os.getcwd() + '/Results_New/'

# Getting data from our modelling
path1 = "/home/jayshil/Documents/Dissertation/ld-project-updated"

# Writing a function so that I do not have to write the same code for 5 times
def spam(path, method):
    name = np.loadtxt(path, usecols=0, unpack=True, dtype=str)
    c1, c2, c3, c4 = np.loadtxt(path, usecols=(1,2,3,4), unpack=True)
    print('Working on method: ', method)
    f1 = open(p1r + 'SPAM_' + method + '.dat', 'w')
    f1.write('#Name\t\t u1_SPAM\t\t u2_SPAM\n')
    for i in tqdm(range(len(name))):
        ab = exoctk.utils.get_target_data(name[i])
        cd = ab[0]
        dict1 = {'transit_duration' : cd['transit_duration'], 
                 'orbital_period' : cd['orbital_period'], 
                 'Rp/Rs' : cd['Rp/Rs'], 
                 'a/Rs' : cd['a/Rs'], 
                 'inclination' : cd['inclination'], 
                 'eccentricity' : 0. if type(cd['eccentricity']) != float else cd['eccentricity'],
                 'omega' : 90. if type(cd['omega']) != float else cd['omega']}
        try:
            xx, yy = tcs(c1[i], c2[i], c3[i], c4[i], planet_data = dict1)
        except:
            print('Well')
            xx, yy = 0.000000123456, 0.000000123456
        f1.write(name[i] + '\t' + str(xx) + '\t' + str(yy) + '\n')
    f1.close()


# juliet LDCs
u1_j, u1_jp, u1_jn, u2_j, u2_jp, u2_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (16,17,18,19,20,21), unpack = True)

# For - Atlas EJ15 LDCs
spam(p1 + '/Atlas/code_us_nl_ata.dat', 'ATLAS-EJ15')

# For - Atlas C17 LDCs
spam(p1 + '/Atlas/claret_us_nl_ata.dat', 'ATLAS-C17')

# For - Phoenix EJ15 LDCs
spam(p1 + '/Phoenix/code_us_nl_pho.dat', 'PHOENIX-EJ15')

# For - Phoenix C17 (q) LDCs
spam(p1 + '/Phoenix/claret_us_nl_pho.dat', 'PHOENIX-C17')

# For - Phoenix C17 (r) LDCs
spam(p1 + '/Phoenix/claret_us_nl_pho_r.dat', 'PHOENIX-C17r')