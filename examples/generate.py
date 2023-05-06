''' Example script generating instances with varying properties. '''
import sys 
sys.path.insert(0,'/home/xinglu/prj/lp-generators-release') 

import numpy as np
from tqdm import tqdm
import pandas as pd
import os, sys 
from numba import njit 

@njit
def set_seed(value):
    np.random.seed(value)
    
from lp_generators.lhs_generators import generate_lhs
from lp_generators.solution_generators import generate_alpha, generate_beta
from lp_generators.instance import EncodedInstance
from lp_generators.features import coeff_features, solution_features
from lp_generators.performance import clp_simplex_performance
from lp_generators.utils import calculate_data, write_instance
from lp_generators.writers import write_tar_encoded, write_mps


@write_instance(write_mps, 'generated/inst_{seed}.mps.gz')
@write_instance(write_tar_encoded, 'generated/inst_{seed}.tar')
@calculate_data(coeff_features, solution_features, clp_simplex_performance)
def generate(seed):
    ''' Generating function taking a seed value and producing an instance
    using the constructor method. The decorators used for this function
    calculate and attach feature and performance data to each generated
    instance, write each instance to a .tar file so they can be loaded
    later to start a search, and write each instance to .mps format. '''
    print('-> ',seed)
    # Seeded random number generator used in all processes.
    np.random.seed(seed)
    set_seed(seed) 
    # Generation parameters to be passed to generating functions.
    size_params = dict(
        variables=np.random.randint(50, 100),
        constraints=np.random.randint(50, 100))
    beta_params = dict(
        basis_split=np.random.uniform(low=0.0, high=1.0))
    alpha_params = dict(
        frac_violations=np.random.uniform(low=0.0, high=1.0),
        beta_param=np.random.lognormal(mean=-0.2, sigma=1.8),
        mean_primal=0,
        std_primal=1,
        mean_dual=0,
        std_dual=1)
    lhs_params = dict(
        density=np.random.uniform(low=0.3, high=0.7),
        pv=np.random.uniform(low=0.0, high=1.0),
        pc=np.random.uniform(low=0.0, high=1.0),
        coeff_loc=np.random.uniform(low=-2.0, high=2.0),
        coeff_scale=np.random.uniform(low=0.1, high=1.0))

    # Run the generating functions to produce encoding components and
    # return the constructed instance.
    instance = EncodedInstance(
        lhs=generate_lhs( **size_params, **lhs_params).todense(),
        alpha=generate_alpha( **size_params, **alpha_params),
        beta=generate_beta( **size_params, **beta_params))
    instance.data = dict(seed=seed)
    return instance

# Create a generator of instance objects from random seeds.
seeds = [
    3072533601, 3601421711, 3085167720, 3845318791, 4240653839,
    956837294, 1416930261, 2883862718, 3309288948, 641946696]
instances = (generate(seed) for seed in seeds)

# Run the generator, keeping only the feature/performance data at each step.
# tqdm is used for progress monitoring.
data = pd.DataFrame([instance.data for instance in tqdm(instances)])

# Print some feature and performance results.
print(data[['seed', 'variables', 'constraints', 'nonzeros', 'clp_primal_iterations']])
