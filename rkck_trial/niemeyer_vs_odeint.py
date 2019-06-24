import numpy as np 
from scipy.integrate import odeint
import time
from sklearn.metrics import mean_squared_error as mse

# number of simulations
THREADS = int(1e2)
repset = int(1e2)

# parameters = [[0.5,0.5, 1.0 + ((i%repset)+1)*1.0/repset] for i in range(THREADS)]
parameters = np.genfromtxt('parameters_3_e2.csv', delimiter = ',')
# parameters = [[1,1,1]] * THREADS

initial_values = [[0.5,0.5,0.5]]*THREADS
NEQN = 3
t_begin = 0.0
step = 10.0
t_end = 100.0
numSteps = int((t_end - t_begin)/step + 1)
time_instants  = [(t_begin + i * step) for i in range(numSteps)]
# definition of the lorenz model for sequential integration
# x' = k0 * (y - x)
# y' = x * (k1 - z) - y
# z' = x * y - k2 * z
def lorenz_model(a, t, k0, k1, k2):
    dxdt = k0*(a[1] - a[0])
    dydt = a[0] * (k1 - a[2]) - a[1]
    dzdt = a[0] * a[1] - k2 * a[2]
    return [dxdt, dydt, dzdt]

def lorenz_jac(a, t, k0, k1, k2):
    J = np.zeros((3,3))
    J[0,0] = - k0
    J[0,1] = k0
    J[1,0] = k1 - a[2]
    J[1,1] = -1
    J[1,2] = -a[0]
    J[2,0] = a[1]
    J[2,1] = a[0]
    J[2,2] = - k2
    return J

# sequential integration results
def serial_sim():
    result = []
    for i in range(THREADS):
        a0 = initial_values[i]
        k = tuple(parameters[i])
        a = odeint(lorenz_model, a0, time_instants, args = k, Dfun = lorenz_jac)
        # a = odeint(lorenz_model, a0, time_instants, args = k)
        result.append(a)

    return result

def parallel_load(filename):
    fil = open(filename, "rb")
    data = np.loadtxt(fil, delimiter = ",")
    print " loaded data of shape " + str(data.shape)
    data = data.reshape((THREADS, numSteps, NEQN))
    print " loaded data converted into shape " + str(data.shape)
    fil.close()
    return data

# to compare accuracy of data
def data_comparison(serial_result, parallel_result):
    total_error = 0
    for thread in range(THREADS):
        total_error += mse(serial_result[thread], parallel_result[thread])

    print "Total mean squared error\t:\t%f"%total_error

# compare the simulations
def sim_comparison():
    serial_start_time = time.time()
    serial_result = serial_sim()
    serial_end_time = time.time()
    print " serial result of length " + str(len(serial_result))
    print " serial result shape threadwise " + str(serial_result[0].shape)
    parallel_start_time = time.time()
    parallel_result = parallel_load('RKCK_parallel_out.csv')

    print "\n\n\n"

    parallel_end_time = time.time()

    print 'Odeint simulations took\t\t:\t%f seconds'%(serial_end_time - serial_start_time)
    print 'Loading data took\t\t\t:\t%f seconds'%(parallel_end_time - parallel_start_time)


    data_comparison(serial_result, parallel_result)



# execute
sim_comparison()
# serial_sim()



