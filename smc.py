import numpy as np
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x

class BoxModel():

    def __init__(self, measurement_array, auxiliary_array, transition_function):
        """
        Parent class for box-model calculations and SMC objects. Use a derived object for calculations.
        In general: 
        N = #state variables
        M = #measurement variables
        L = #datapoints (time-axis)
        N_U = #auxiliary variables
        params:
        measurement_array: numpy array of shape L x M (=L x N)
        auxiliary_array: numpy array of shape L x N_U or None
        transition function: function that performs calculations to propagate the state from timestep index-1 to timestep index. 
          expects params (p, u) with
          p = numpy array of shape N
          u = numpy array of shape N_U or None
          and returns a numpy array of shape N
        """
        self.z = measurement_array
        self.u = auxiliary_array
        self.transition_function = transition_function
        self.l, self.m = self.z.shape
        self.n = self.m
        self.t = 0
        self.initialized = False

    def generate_p0(self):
        """
        Generate initial state variables. 
        returns the measurement variables at time self.t
        """
        return self.z[self.t]

    def get_auxiliary(self, index):
        """
        Wrapper function to obtain u at index, if given. Returns None if u is None
        """
        if self.u is None:
            return None
        return self.u[index]
    
    def iterate(self, start, end, verbose=False):
        """
        Iterate over the time-axis L from start to end. Displays a tqdm progress bar if verbose is True. 
        At each iteration, set the time self.t, apply the calculations and write the output values.
        """
        iter = range(start, end,)
        if verbose:
            iter = tqdm(iter)
        for i in iter:
            self.t = i
            self.p = self.step(i)
            self.write()
        pass

    def step(self,index):
        """
        Iteration step. Performs the calculation for index and initializes self if not done yet.
        returns a numpy array of shape N
        """
        if not self.initialized:
            self.initialize()
        return self._step(index)
        
    def initialize(self):
        """
        Initialization: set time to 0, generate state vector and define output arrays.
        """
        self.t = 0 
        self.x = np.zeros((self.l, self.n))
        self.p = self.generate_p0()
        self.initialized = True

    def _step(self, index):
        """
        Inner iteration function at index.
        returns a numpy array of shape N
        """
        return self.transition(self.p, self.get_auxiliary(index))
    
    def transition(self, p, u):
        """
        Transition function wrapper: Calls transition_function(p, u)
        returns a numpy array of shape N 
        """
        return self.transition_function(p, u)
    
    def write(self):
        """
        writes current state self.p to output of index self.t
        """
        self.x[self.t] = self.p

    def __call__(self, verbose=False):
        """
        Iterate over the time-axis L. Displays a tqdm progress bar if verbose is True. 
        Returns the output state vector x: numpy array of shape L x N
        """
        self.iterate(0, self.l, verbose=verbose)
        return self.x
    pass


class ConstrainedBoxModel(BoxModel):

    def __init__(self, measurement_array, auxiliary_array, transition_function, constrained):
        """
        Constrained box-model object. Performs calculations according to transition_function and resets constrained variables to the current measurement.
        In general: 
        N = #state variables
        M = #measurement variables
        L = #datapoints (time-axis)
        N_U = #auxiliary variables
        params:
        measurement_array: numpy array of shape L x N. Unmeasured or unconstrained variables have to be provided via all-zero columns.
        auxiliary_array: numpy array of shape L x N_U or None
        transition function: function that performs calculations to propagate the state from timestep index-1 to timestep index.
          expects params (p, u) with
          p = numpy array of shape N
          u = numpy array of shape N_U or None
          and returns a numpy array of shape N
        constrained: numpy array of shape N with 1s indicating which variables are to be constrained by the measurement and 0s indicating which variables will not be constrained by measurement. 
        """
        super().__init__(measurement_array, auxiliary_array, transition_function,)
        self.constrained = constrained
    
    def _step(self, index):
        """
        Constrained inner iteration function at index. Calculates new state variables via transition function but overrides constrained variables by measurement at timestep index.
        returns a numpy array of shape N
        """
        p_measurement = self.generate_p0()
        p_transition = self.transition(self.p, self.get_auxiliary(index))
        constrained_index = np.where(np.logical_and(self.constrained, np.isfinite(p_measurement)))
        p_transition[constrained_index] = p_measurement[constrained_index]
        return p_transition
    

class AuxParticleFilter(BoxModel):
    
    def __init__(self,measurement_array, auxiliary_array, transition_function, precision_array, detlim_array, sigma_0_const, sigma_0_rel, loglikeli_function=None, randomize_function=None,generate_particles_function=None, num_particles=1000, num_aux_particles=10000):
        """
        SMC auxiliary particle filter object. Spawns ensembles of state variables and propagates ensembles according to stochastic processes described by sigma_0_const and sigma_0_rel and deterministic processes given by transition_function. Resamples from ensemble proportional to the likelihood with precision_array and detlim_array.
        In general: 
        N = #state variables
        M = #measurement variables
        L = #datapoints (time-axis)
        N_U = #auxiliary variables
        K = #particles
        R = #auxiliary particles
        params:
        measurement_array: numpy array of shape L x M
        auxiliary_array: numpy array of shape L x N_U or None
        transition function: function that performs calculations to propagate the state from timestep index-1 to timestep index.
          expects params (p, u) with
          p = numpy array of shape K x N
          u = numpy array of shape N_U or None
          and returns a numpy array of shape K x N
        precision_array: numpy array of shape M containing the relative uncertainty (1sigma) of each measurement
        detlim_array: numpy array of shape M containing the constant uncertainty (1sigma) of each measurement
        sigma_0_const: numpy array of shape N containing the constant subjective variability of each state variable
        sigma_0_const: numpy array of shape N containing the relative subjective variability of each state variable
        
        loglikeli_function: function that returns the loglikelihood of each particle accoring to the measurement at timestep index; or None
          expects params (p, y, u) with
          p = numpy array of shape K x N
          y = numpy array of shape M
          u = numpy array of shape N_U or None
          and returns a numpy array of shape K
         if None, assumes N = M, each index along N matches each index along M, and Gaussian error for each measurement.
         
        randomize_function: function that performs the stochastic propagation from timestep index-1 to timestep index; or None
          expects params (p, y, u) with
          p = numpy array of shape K x N
          u = numpy array of shape N_U or None
          and returns a numpy array of shape K x N
         if None, assumes lognormal distribution with standard deviation^2 = sigma_0_const^2 + p^2 sigma_0_rel^2
        
        generate_particles_function: function that generates initial particle ensemble; or None
          expects params (p, y, u) with
          p = numpy array of shape K x N
          u = numpy array of shape N_U or None
          and returns a numpy array of shape K x N
         if None, assumes N = M, each index along N matches each index along M, and generates initial particles by calling randomize after setting each particle to the current measurement 
        num_particles: K
        num_aux_particles: R 
        """
        super().__init__(measurement_array, auxiliary_array, transition_function,)
        self.precision = precision_array
        self.detlim = detlim_array
        self.sigma_0_const = sigma_0_const
        self.sigma_0_rel = sigma_0_rel
        self.k = num_particles
        self.r = num_aux_particles
        self.loglikeli_function = loglikeli_function
        self.randomize_function = randomize_function
        self.generate_particles_function = generate_particles_function
        self.entropy = np.zeros((self.l))
        self.aux_entropy = np.zeros(self.l)
        self.run_entropy = np.zeros((self.l))
        pass
        
    def initialize(self):
        """
        Initialization: set time to 0, generate state vector and define output arrays.
        """
        self.p = self.generate_p0()
        self.n = self.p.shape[1]
        self.x = np.zeros((self.l, self.n))
        self.sx = np.zeros((self.l, self.n))
        self.initialized = True

    def generate_p0(self):
        """
        Generate initial state variables. 
        Uses generate_particles_function if given; otherwise assumes N = M, each index along N matches each index along M, and generates initial particles by calling randomize after setting each particle to the current measurement 
        returns numpy array of shape K x N
        """
        if self.generate_particles_function is None: 
            p0_mu = super().generate_p0()
            p0_var = np.repeat(p0_mu[None,:], self.k, 0)
            return AuxParticleFilter.randomize(self, p0_var, self.get_auxiliary(self.t))
        return self.generate_particles_function(self)
    
    def randomize(self, p, u):
        """
        performs the stochastic propagation from timestep index-1 to timestep index.
        params:
        p = numpy array of shape K x N
        u = numpy array of shape N_U or None
        Calls randomize_function(p, u) if given; otherwise assumes lognormal distribution with standard deviation^2 = sigma_0_const^2 + p^2 sigma_0_rel^2
        returns numpy array of shape K x N
        """
        if self.randomize_function is None:
            s = np.sqrt(self.sigma_0_const[None,:]**2 + (self.sigma_0_rel[None,:] * p)**2)
            p_var = np.maximum(p, s/100.) 
            return np.random.lognormal(np.log(np.square(p_var)/np.sqrt(np.square(p_var)+np.square(s))),np.sqrt(np.log(1 + s**2/(p_var**2))))
        return self.randomize_function(p, u)
    
    def meas_loglikeli(self, p, y, u):
        """
        returns the loglikelihood of each particle accoring to the measurement at timestep index
        params:
        p = numpy array of shape K x N
        y = numpy array of shape M
        u = numpy array of shape N_U or None
        Calls loglikeli_function if given; otherwise assumes N = M, each index along N matches each index along M, and Gaussian error for each measurement.
        returns numpy array of shape K
        """
        if self.loglikeli_function is None:
            s = (self.detlim**2 + (self.precision * y)**2)[None,:]
            probs = -np.square(p - y[None,:])/2/s
            probs[np.where(np.isnan(probs))] = 0
            return np.sum(probs,axis=1)
        return self.loglikeli_function(p, y, u)
    
    def _step(self, index):
        """
        SMC inner iteration function at index. Propagates particles via transition function, calculates auxiliary weights, resamples R auxiliary particles according to these weights, propagates auxiliary particles via stochastic randomization and deterministic transition, calculates new weights, rescales by auxiliary weights, resamples from auxiliary particles according to these weights. 
        returns a numpy array of shape N
        """
        y = self.z[index]
        u = self.get_auxiliary(index)
        #auxiliary phase
        #all particles
        indices_sample = np.arange(self.k)
        #propagate using only deterministic part of prior
        particles_sample =  self.prior(indices_sample, aux=True)
        #calculate auxiliary weights
        aux_llhood = self.meas_loglikeli(particles_sample, y, u)
        aux_weights, aux_cdf, self.aux_entropy[index] = self.weights(aux_llhood)
        #choose R particles according to auxiliary weights
        u_bootstrap = np.random.uniform(0,1,self.r)
        indices_run = np.searchsorted(aux_cdf, u_bootstrap)
        #propagate using full description of prior
        particles_run = self.prior(indices_run)
        #calculate weights
        run_llhood = self.meas_loglikeli(particles_run, y, u)
        run_weights, run_cdf, self.run_entropy[index] = self.weights(run_llhood)
        #reweight
        llhood = run_llhood - aux_llhood[indices_run]  
        weights, cdf, self.entropy[index] = self.weights(llhood)
        #resample from auxiliary particles
        u_run = np.random.uniform(0,1,self.k)
        indices_post = np.searchsorted(cdf, u_run)
        return particles_run[indices_post]
       
    def write(self):
        """
        writes mean and standard deviation of current state vector self.p to x and sx at index self.t
        """
        self.x[self.t] = np.mean(self.p, axis=0)
        self.sx[self.t] = np.std(self.p, axis=0, ddof=1)
        
    
    def weights(self, llhood):
        """
        Helper function that converts log-likelihoods to units of relative probability, cumulative probability and entropy.
        """
        #against overflow
        weights = llhood - np.nanmax(llhood)
        weights = np.exp(weights)
        #relative probability
        weights /= np.nansum(weights)
        #against underflow
        entropy = -np.nansum(np.maximum(weights,1e-15) * np.log(np.maximum(weights,1e-15)))
        #in case of nans
        cdf = np.nancumsum(weights)
        cdf /= cdf[-1]
        return weights, cdf, entropy
   
    def prior(self, indices, aux = False):
        """
        Combined propagation function that randomizes the particles at p[indices] if aux is False and calculates the deterministic transition afterwards. 
        indices is an array of shape J = R or K
        returns numpy array of shape J x N 
        """
        particles = self.p[indices]
        u = self.get_auxiliary(self.t)
        if not aux:
            particles = self.randomize(particles, u)
        particles = self.transition(particles, u)
        return particles

class ActiveAuxParticleFilter(AuxParticleFilter):

    def __init__(self, *args, switch_prob=0.025, **kwargs):
        """
        Creates an AuxParticleFilter(*args, **kwargs) object with an additional activity state variable in {0, 1}. Initial particles are spawned 50% active and 50% passive, passive particles only follow the stochastic part of the prior. 
        Each particle switches its state during randomization with probability switch_prob.
        """
        super().__init__(*args, **kwargs)
        self.p_eta = switch_prob
        pass

    def generate_p0(self):
        p0_var = super().generate_p0()
        p0_activity = np.zeros((self.k, 1), dtype=int)
        p0_activity[self.k//2:] = 1
        return np.concatenate([p0_var, p0_activity], axis=1)
    
    def randomize(self, p, u):
        p_var = super().randomize(p[:,:-1], u)
        p_activity = p[:,-1:]
        do_switch = np.greater(np.random.uniform(size=p.shape[0]),1-self.p_eta)[:,None]
        p_activity = p_activity + do_switch * (1 - 2 * p_activity)
        return np.concatenate([p_var, p_activity],axis=1)
    
    def meas_loglikeli(self, p, y, u):
        return super().meas_loglikeli(p[:,:-1], y, u)
    
    def transition(self, p, u):
        p_passive = p[:,:-1]
        p_active = super().transition(p_passive, u)
        p_mask = p[:,-1:]
        return np.concatenate([p_passive + p_mask * (p_active - p_passive), p_mask], axis=1)

    pass

## Example transition functions

#transition function for AuxParticleFilter
def transition_leighton_smc(p, u):
    #u = p [hPa], T [°C]
    # reaction rate [ppbv^-1 s^-1] 
    k_chem =  1.9e-6 * (u[0] * 100) / (273.15 + u[1]) / 8.314 * 6.022
    newp = p.copy()
    # O3, NO, NO2, jNO2
    for i in range(60): # dt(chem) = 1s, dt(data) = 1min
        r1 = newp[:,3] * newp[:,2]
        r2 = k_chem *  newp[:,0] * newp[:,1]
        newp[:,0] += (r1 - r2)
        newp[:,1] += (r1 - r2)
        newp[:,2] += (r2 - r1)
    return newp

#transition function for BoxModel
def transition_leighton_box(p, u):
    #u = p [hPa], T [°C]
    # reaction rate [ppbv^-1 s^-1]
    k_chem =  1.9e-6 * (u[0] * 100) / (273.15 + u[1]) / 8.314 * 6.022
    newp = p.copy()
    # O3, NO, NO2, jNO2
    for i in range(60): # dt(chem) = 1s, dt(data) = 1min
        r1 = newp[3] * newp[2]
        r2 = k_chem *  newp[0] * newp[1]
        newp[0] += (r1 - r2)
        newp[1] += (r1 - r2)
        newp[2] += (r2 - r1)
    return newp