import numpy as np 


NPAR = 3
NSAMPLES = int(1e1)

# range for sampling
lb = [0.5 for i in range(NPAR)]
ub = [2.0 for i in range(NPAR)]

margin = [ub[i] - lb[i] for i in range(NPAR)]

print("Sample requested:\t%d SAMPLES FOR\t%d PARAMETERS"%(NSAMPLES, NPAR))

parameters = np.random.rand(NSAMPLES, NPAR)
for i in range(NPAR):
	parameters[:, i] = lb[i] + margin[i] * parameters[:, i]

# parameters = [[10.0,28.0, 8.0/3.0] for i in range(NSAMPLES)]

print("Dumping parameters")
np.savetxt("parameters.csv", parameters, delimiter = ",")
