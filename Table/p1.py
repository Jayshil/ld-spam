import numpy as np

path1 = '/home/jayshil/Documents/Dissertation/ld-spam/Results_New/'
path2 = '/home/jayshil/Documents/Dissertation/ld-spam'

name = np.loadtxt(path1 + 'SPAM_ATLAS-EJ15.dat', usecols = 0, unpack = True, dtype=str)

u1_code_ata, u2_code_ata = np.loadtxt(path1 + 'SPAM_ATLAS-EJ15.dat', usecols = (1,2), unpack = True)
u1_cla_ata, u2_cla_ata = np.loadtxt(path1 + 'SPAM_ATLAS-C17.dat', usecols = (1,2), unpack = True)

u1_code_pho, u2_code_pho = np.loadtxt(path1 + 'SPAM_PHOENIX-EJ15.dat', usecols = (1,2), unpack = True)
u1_cla_pho, u2_cla_pho = np.loadtxt(path1 + 'SPAM_PHOENIX-C17.dat', usecols = (1,2), unpack = True)
u1_cla_pho_r, u2_cla_pho_r = np.loadtxt(path1 + 'SPAM_PHOENIX-C17r.dat', usecols = (1,2), unpack = True)


table1 = open(path2 + '/Table/ldc.dat','w')

for i in range(len(name)):
	table1.write(name[i][:-1] + ' & ' + str(np.format_float_positional(u1_code_ata[i], 2)) + ' & ' + str(np.format_float_positional(u2_code_ata[i], 2)) + ' & ' + str(np.format_float_positional(u1_code_pho[i], 2)) + ' & ' + str(np.format_float_positional(u2_code_pho[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho_r[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho_r[i], 2)) + ' \\\\ \n\t')

table1.close()