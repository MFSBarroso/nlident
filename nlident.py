"""
Toolkit for System Identification for NARX models
with discrete-time polynomial representation.

Prof. Dr. Márcio Falcão Santos Barros (DEPEL - UFSJ)
MEE. Jim Jones da Silveira Marciano

Second semester of 2024.
"""

#importing libraries
#==============================================================================
from numpy import (round, any, vstack, zeros, array, max, arange, hstack,
                   prod, concatenate, fliplr, ones, roll,
                   floor, where, ndarray, sum, sqrt, log,
                   mean, integer, var, newaxis, argmin, abs, pi, std, isnan)
from itertools import combinations_with_replacement as combinations
from numpy.linalg import lstsq
from matplotlib.pyplot import (figure, plot, axhline, grid, xlabel,
                               ylabel, title, legend, show)
#==============================================================================

#==============================================================================
def genterms(*args):
  """
    Parameters:
      The function parameters follow this order:
      gnl   (int): Degree of nonlinearity. (Mandatory)
      lagy  (int): Maximum output lag      (Mandatory)
      lagu  (int): Maximum input lag       (Optional)
      lage  (int): Maximum error lag       (Optional)
    Returns:
      Model  - Matrix representation of the model
      Tterms - Total number of terms

  """

  #Variable Initialization
  ulin = array([])
  ylin = array([])
  elin = array([])

  if len(args) < 2:
    print('the function needs at least 2 parameters')
    return None, None
  gnl = args[0]
  lagy = args[1]
  lagu = None
  lage = None
  if len(args) == 3:
     lagu = args[2]
  if len(args) == 4:
     lagu = args[2]
     lage = args[3]

  if not isinstance(gnl, (int)):
     print('gnl must be an integer')
     return None, None
  if gnl < 1:
     print('gnl must be greater than zero')
     return None, None
  if not isinstance(lagy, (int)):
     print('lagy must be an integer')
     return None, None
  if lagy < 1:
     print('lagy must be greater than zero')
     return None, None
  if lagu:
     if not isinstance(lagu, (int)):
        print('lagu must be an integer')
        return None, None
     if lagu < 1:
        print('lagu must be greater than zero')
        return None, None
     #Create linear input terms
     ulin = arange(2001,2001+lagu)
  if lage:
     if not isinstance(lage, (int)):
        print('lage must be an integer')
        return None, None
     if lage < 1:
        print('lage must be greater than zero')
        return None, None
     elin = arange(3001,3001+lage)

  #Create linear output terms
  ylin = arange(1001,1001+lagy)
  #Concatenating ylin, ulin, and elin vectors to generate the model matrix
  Flin = concatenate((array([0]), ylin[:]))
  if len(args) == 3:
    Flin = concatenate((Flin, ulin[:]))
  if len(args) == 4:
    Flin = concatenate((Flin, ulin[:], elin[:]))

  #Create combinations of linear terms to generate all process and noise terms
  model = fliplr(array(list(combinations(Flin, gnl))))
  Tterms = len(model)
  return model, Tterms
#==============================================================================

#==============================================================================
def build_pr(*args):
    """
      Parameters:
        The function parameters follow this order:
        model  (array): Set of candidate terms - Process. (Mandatory)
        u      (array): input data             (Mandatory)
        y      (array): output data            (Mandatory)

      Returns:
        P (array): Process regressor matrix

    """

    #Check the number of function inputs
    #--------------------------------------------------------------------------
    if len(args) != 3:
        print("The 'build_pr' function requires 3 arguments: model, u, and y")
        return None

    #Assigning parameter values to variables model, u, y
    #--------------------------------------------------------------------------
    model = args[0]
    u     = args[1]
    y     = args[2]
    #Check if inputs are integers
    #--------------------------------------------------------------------------
    if type(u) == int or type(y) == int:
        print("u and y must be vectors")
        return None
    else:
        n = len(u)
    #--------------------------------------------------------------------------
    #Check if model is a vector
    #--------------------------------------------------------------------------
    if type(model) == int:
        print("model must be a matrix")
        return None
    else:
        l, c = model.shape
    #--------------------------------------------------------------------------

    P = ones((n, l))

    for i in range(l):
        #determine if it is input or output (1 is output, 2 is input, and 0 is constant)
        #--------------------------------------------------------------------------
        tipo = floor(model[i, :] / 1000).astype(int)  # Signal
        #--------------------------------------------------------------------------
        #Determine the delays contained in the model
        #--------------------------------------------------------------------------
        delay = model[i, :] - tipo * 1000
        #--------------------------------------------------------------------------
        #j will receive the index where there are only output data
        #--------------------------------------------------------------------------
        j = where(tipo == 1)[0]
        if j.size > 0:
            for k in range(len(j)):
                #Perform the product of the regressor matrix where the elements are
                #the outputs y
                #------------------------------------------------------------------
                P[:, i] *= roll(y, int(delay[j[k]]))
                #------------------------------------------------------------------
        #j will receive the index where there are only input data
        #--------------------------------------------------------------------------
        j = where(tipo == 2)[0]
        if j.size > 0:
            for k in range(len(j)):
                #Perform the product of the regressor matrix where the elements are
                #the inputs u
                #------------------------------------------------------------------
                P[:, i] *= roll(u, int(delay[j[k]]))
                #------------------------------------------------------------------
        #--------------------------------------------------------------------------
    return P
#==============================================================================

#==============================================================================
def build_no(*args):
    """
      Parameters:
        The function parameters follow this order:
        model  (array): Set of candidate terms - Process. (Mandatory)
        u      (array): input data             (Mandatory)
        y      (array): output data            (Mandatory)
        e      (array): noise data             (Mandatory)

      Returns:
        Pno (array): Noise regressor matrix

    """

    #Check the number of function inputs
    #--------------------------------------------------------------------------
    if len(args) != 4:
        print("The 'build_no' function requires 4 arguments: model, u, y, and e")
        return None

    #Assigning parameter values to variables model, u, y, e
    #--------------------------------------------------------------------------
    model = args[0]
    u     = args[1]
    y     = args[2]
    e     = args[3]
    #Check if inputs are integers
    #--------------------------------------------------------------------------
    if type(u) == int or type(y) == int or type(e) == int:
        print("u, y, and e must be vectors")
        return None
    else:
        n = len(u)
    #--------------------------------------------------------------------------
    #Check if model is a vector
    #--------------------------------------------------------------------------
    if type(model) == int:
        print("model must be a matrix")
        return None
    else:
        l, c = model.shape
    #--------------------------------------------------------------------------
    P = ones((n, l))
    
    for i in range(l):
        #determine if it is input or output (1 is output, 2 is input, and 0 is constant)
        #--------------------------------------------------------------------------
        tipo = floor(model[i, :] / 1000).astype(int)  # Signal
        #--------------------------------------------------------------------------
        #Determine the delays contained in the model
        #--------------------------------------------------------------------------
        delay = model[i, :] - tipo * 1000
        #--------------------------------------------------------------------------
        #j will receive the index where there are only output data
        #--------------------------------------------------------------------------
        j = where(tipo == 1)[0]
        if j.size > 0:
            for k in range(len(j)):
                #Perform the product of the regressor matrix where the elements are
                #the outputs y
                #------------------------------------------------------------------
                P[:, i] *= roll(y, int(delay[j[k]]))
                #------------------------------------------------------------------
        #j will receive the index where there are only input data
        #--------------------------------------------------------------------------
        j = where(tipo == 2)[0]
        if j.size > 0:
            for k in range(len(j)):
                #Perform the product of the regressor matrix where the elements are
                #the inputs u
                #------------------------------------------------------------------
                P[:, i] *= roll(u, int(delay[j[k]]))
                #------------------------------------------------------------------
        #--------------------------------------------------------------------------
        #j will receive the index where there are only noise data
        #--------------------------------------------------------------------------
        j = where(tipo == 3)[0]
        if j.size > 0:
            for k in range(len(j)):
                #Perform the product of the regressor matrix where the elements are
                #noise
                #------------------------------------------------------------------
                P[:, i] *= roll(e, int(delay[j[k]]))
                #------------------------------------------------------------------
        #--------------------------------------------------------------------------
    Pno = P
    return Pno
#==============================================================================
def get_info(modelo):
    """
      Parameters:
        The function parameters follow this order:
          model  (array): Set of candidate terms. (Mandatory)
      Returns:
        model_pr (array): Set of candidate terms - Process.
              ntp   (int): number of process terms
              gnl   (int): degree of nonlinearity
              nu    (int): maximum lag of u
              ny    (int): maximum lag of y
              mlag  (int): maximum lag of the model
    """
    #Get the number of rows and columns of the candidate terms matrix
    #---------------------------------------------------------------------------    
    l, c = modelo.shape
    #---------------------------------------------------------------------------
    
    #Check which regressors are present: input, output, and noise
    #---------------------------------------------------------------------------
    tipo = zeros(l, dtype=int)
    for i in range(l):
        tipo[i] = round(modelo[i, 0] / 1000).astype(int)
    #--------------------------------------------------------------------------- 
   
    #Calculate the number of process terms, noise terms, and degree of nonlinearity, in this order
    #---------------------------------------------------------------------------
    ntp = len(where(tipo < 3)[0])
    ntr = l - ntp
    gnl = c
    #---------------------------------------------------------------------------
    
    #Initial conditions for the candidate terms matrices of process and noise
    #---------------------------------------------------------------------------
    modelo_pr = zeros((l, c), dtype=int)
    modelo_no = zeros((l, c), dtype=int)
    #---------------------------------------------------------------------------

    #Build the process regressor matrices
    #---------------------------------------------------------------------------
    for i in range(l):
        for j in range(c):
            if 1000 < modelo[i, 0] < 2000:
                modelo_pr[i, j] = modelo[i, j]
            elif 2000 < modelo[i, 0] < 3000:
                modelo_pr[i, j] = modelo[i, j]
            elif modelo[i, 0] > 3000:
                modelo_no[i, j] = modelo[i, j]
    #Build the process regressor matrix, cutting rows with zeros except the row with the constant
    #---------------------------------------------------------------------------
    modelo_no = modelo_no[any(modelo_no != 0, axis=1)]
  
    #---------------------------------------------------------------------------
    
    if modelo_no.size == 0:
        ne = 0 
    else:
        #Build the process regressor matrices
        #---------------------------------------------------------------------------
        l, c = modelo_no.shape
        atrasoe = zeros((l, c), dtype=int)
        for i in range(l):
            for j in range(c):
                if modelo_no[i, j] > 3000:
                    atrasoe[i, j] = modelo_no[i, j] - 3000
        ne = max(atrasoe)
        #---------------------------------------------------------------------------
   
    #Build the process regressor matrix, cutting rows with zeros except the row with the constant
    #---------------------------------------------------------------------------
    modelo_pr = modelo_pr[any(modelo_pr != 0, axis=1)]
    complemento = zeros((1,c))
    modelo_pr = vstack((complemento, modelo_pr))
    #---------------------------------------------------------------------------
    
    
    l, c = modelo_pr.shape
    atrasou = zeros((l, c), dtype=int)
    atrasoy = zeros((l, c), dtype=int)
    
    #Build a vector of input and output lags for each row and column of the candidate process terms matrix
    #---------------------------------------------------------------------------
    for i in range(l):
        for j in range(c):
            if 1000 < modelo_pr[i, j] < 2000:
                atrasoy[i, j] = modelo_pr[i, j] - 1000
            elif modelo_pr[i,j] > 2000:
                atrasou[i, j] = modelo_pr[i, j] - 2000
    
    #Calculate the maximum overall lag of the model variable
    #---------------------------------------------------------------------------
    ny = max(atrasoy)
    nu = max(atrasou)
    mlag = max(array([nu, ny, ne]))
    #---------------------------------------------------------------------------
    
    return modelo_pr, modelo_no, ntp, ntr, gnl, nu, ny, ne, mlag
#==============================================================================

#==============================================================================
def mcand(model, cy, cu):
    """
    Returns the modified set of candidate terms after excluding 
    terms that belong to a pre-selected cluster.

    Input Parameters:
        model - matrix of candidate terms
        cy, cu - indicates the cluster to be excluded 

    Returns:
       nmodel - the new set of candidate terms
    """
    if not isinstance(cy, int) or cy < 0:
        raise ValueError("Error: The parameter 'cy' must be a positive integer.")
        return
    if not isinstance(cu, int) or cu < 0:
        raise ValueError("Error: The parameter 'cu' must be a positive integer.")
        return
    if len(model.shape) != 2:
        raise ValueError('Cand must be a 2D matrix.')
    
    toterms, degree = model.shape

    nmodel = []
    for i in range(toterms):
        auxy = 0
        auxu = 0
        auxe = 0
        for j in range(degree):
            a = model[i, j]
            kk = a // 1000
            if kk == 1:
                auxy += 1
            elif kk == 2:
                auxu += 1
            elif kk == 3:
                auxe += 1
        
        if auxe > 0:
            nmodel.append(model[i, :])
        elif auxe == 0 and (auxy != cy or auxu != cu):
            nmodel.append(model[i, :])
    
    nmodel = array(nmodel)
    
    if nmodel.shape[0] == toterms:
        raise ValueError('The specified cluster does not exist in the model.')
    return nmodel
#==============================================================================

#==============================================================================
def sort_pr(*args):
    """
    Function that sorts the model based on the variance of the generated residuals.

    Input parameters:
        - model_pr: Matrix containing the prediction model.
        - u: Vector of system inputs.
        - y: Vector of system outputs (measured real values).

    Output parameters:
        - model: Matrix reordered according to the variances of the residuals.
        - variance: Matrix containing the variance of the residuals and the original index of the terms, sorted by variance.
        - residuals: Matrix of residuals calculated between the real values (y) and the estimated values (yhat).
    """

    # Unpacking the provided arguments
    model_pr, u, y = args

    # The four ready functions: genterms, get_info, build_pr, and build_no
    #-------------------------------------------------------------------------------
    P = build_pr(model_pr, u, y)
    #------------------------------------------------------------------------------
    l, c = P.shape

    yhat = zeros((len(y), 0))
    residuals = zeros((len(y), c))
    variance = zeros((c, 2))

    for l in range(c):
        Parameters = lstsq(P[:, l].reshape(-1, 1), y[:, newaxis], rcond=None)[0]
        yhat = P[:, l] * Parameters
        residuals[:, l] = y - yhat
        variance[l, :] = [var(residuals[:, l]), l]

    variance = variance[variance[:, 0].argsort()]
    order = variance[:, 1].astype(int)
    model = model_pr[order, :]
    #------------------------------------------------------------------------------

    return model, variance, residuals
#==============================================================================

#==============================================================================
def sort_no(*args):
    """
    Function that sorts the model based on the variance of the generated residuals.

    Input parameters:
        - model_pr: Matrix containing the prediction model.
        - u: Vector of system inputs.
        - y: Vector of system outputs (measured real values).

    Output parameters:
        - model: Matrix reordered according to the variances of the residuals.
        - variance: Matrix containing the variance of the residuals and the original index of the terms, sorted by variance.
        - residuals: Matrix of residuals calculated between the real values (y) and the estimated values (yhat).
    """

    # Unpacking the provided arguments
    model_no, u, y, e = args

    # The four ready functions: genterms, get_info, build_pr, and build_no
    #-------------------------------------------------------------------------------
    P = build_no(model_no, u, y, e)
    #------------------------------------------------------------------------------
    l, c = P.shape

    yhat = zeros((len(y), 0))
    residuals = zeros((len(y), c))
    variance = zeros((c, 2))

    for l in range(c):
        Parameters = lstsq(P[:, l].reshape(-1, 1), y[:, newaxis], rcond=None)[0]
        yhat = P[:, l] * Parameters
        residuals[:, l] = y - yhat
        variance[l, :] = [var(residuals[:, l]), l]

    variance = variance[variance[:, 0].argsort()]
    order = variance[:, 1].astype(int)
    model = model_no[order, :]
    #------------------------------------------------------------------------------

    return model, variance, residuals
#==============================================================================

#==============================================================================
def simodeld(model, Parameters, uv, y0):
    """
    Simulates the dynamic model using the provided parameters and input data.

    Input parameters:
    - model: ndarray (2D array)
        Matrix containing the model structure, where each row represents a prediction term
        and each column represents the delays and types of variables (input/output).
    - Parameters: ndarray (1D array)
        Vector containing the estimated parameters for each term of the model.
    - uv: ndarray (1D array)
        Vector of system inputs (u) over time.
    - y0: float (1d array)
        Initial value of the system output (initial condition).

    Output:
    - yhat: ndarray (1D array)
        Vector of estimated outputs by the model for all time samples based on the inputs (uv) and parameters (Parameters).
    """
    modelo_pr, modelo_no, ntp, ntr, gnl, nu, ny, ne, mlag = get_info(model)
    
    rows, cols = model.shape
    l, c = uv[:, newaxis].shape
    yhat = zeros(l)
    yhat[:mlag] = y0
    yhat = yhat[:, newaxis]


    for i in range(mlag, l):
        signal = zeros(rows)
        for j in range(rows):
            PL = ones(cols).astype(float)
            tipo = floor(model[j, :] / 1000).astype(int)
            delay = model[j, :] - tipo * 1000
            k = where(tipo == 1)[0]
            if k.size > 0:
                for L in range(len(k)):
                    PL[L] *= yhat[i - delay[L].astype(int)]
            k = where(tipo == 2)[0]
            if k.size > 0:
                for L in range(len(k)):
                    PL[L] *= uv[i - delay[L].astype(int)]
            signal[j] = Parameters[j].dot(prod(PL))
        yhat[i] = sum(signal)
    return yhat
#==============================================================================

#==============================================================================
def ols(model, ui, yi, ntp):
    """
    Ordinary Least Squares

    Input parameters:
    - model: ndarray (2D array)
        Matrix containing the model structure, where each row represents a prediction term
        and each column represents the delays and types of variables (input/output).
    - ui: ndarray (1D array)
        Vector of system identification inputs over time.
    - yi: ndarray (1D array)
        Vector of system identification outputs over time.
    - ntp: int
        Number of process terms to be used

    Output:
    - Parameters: ndarray (1D array)
        Vector of parameters estimated by the ols
    - model: new model with the number of process terms used

    """

    model = model[0:ntp, :]
    P = build_pr(model, ui, yi)
    Parameters = lstsq(P, yi[:, newaxis], rcond=None)[0]

    return Parameters, model
#==============================================================================

#==============================================================================
def els(model, ntp, ntr, nir, u, y):
    '''
    The ELS function implements the extended least squares method, with inputs:
        
        model: matrix of candidate terms with
           ntp: number of process terms
           ntr: number of noise terms
           nir: number of noise iterations
             u: input data
             y: output data
             
    Outputs:
    
        Parameters: mean of the nir noise iterations
          Variance: variance of the nir noise iterations
        ProcessPar: mean of the nir noise iterations of the Process Parameters
          NoisePar: mean of the nir noise iterations of the Noise Parameters
    '''
    
    # From the model, get_info separates the process and noise models
    model_pr, model_no, nt, nr, gnl, nu, ny, ne, mlag = get_info(model)
    # Using sort_pr, it's possible to sort candidate terms based on the 
    # explanation of the estimation error variance
    model_pr, variance, residuals = sort_pr(model_pr, u, y)
       
    # Split the models into process and noise with the user-defined size
    if ntr >= ne:
        ntr = ne
    model_no = model_no[:ntr]  
    if ntp >= nt:
        ntp = nt
    model_pr = model_pr[:ntp]
    
    
    # Fill the regressor matrix for the process model for the first iteration
    P_pr = build_pr(model_pr, u, y)
    
    # Initialize the Parameters matrix    
    M_Parameters = zeros((ntp + ntr, nir + 1))
    
    
    # Start the user-defined number of noise iterations
    for i in range(nir):
        if i == 0: # First iteration generates the first noise signal
            P = build_pr(model_pr, u, y)
            Par = lstsq(P, y[:, newaxis], rcond=None)[0]
            yhat = P @ Par
            e = (yhat.T - y.T)[0]
            model_no, variance, residuals = sort_no(model_no, u, y, e)
        else: # Start noise iterations
            P_pr = build_pr(model_pr, u, y)
            P_no = build_no(model_no, u, y, e) 
            P = hstack((P_pr, P_no))
            Parameters = lstsq(P, y[:, newaxis], rcond=None)[0]
            yhat = P_pr @ Parameters[:ntp]
            e = (yhat.T - y.T)[0]
            Parameters = Parameters.squeeze()
            M_Parameters[:, i+1] = Parameters
            
    # Calculate mean and variance of Parameters   
    Parameters = mean(M_Parameters, axis=1)
    Variance = var(M_Parameters, axis=1)
    ParProcess = Parameters[0:ntp, newaxis]
    ParNoise = Parameters[ntp:, newaxis]
    residue = e
    
    return Parameters, Variance, residue, model_pr, model_no, ParProcess, ParNoise, M_Parameters
#==============================================================================

#==============================================================================
def coefcorr(*args):
    """
    Calculates the correlation coefficient between two vector variables.

    Parameters (via *args):
    1. vet1 -- data vector (numpy array)
    2. vet2 -- data vector (numpy array)

    Returns:
    r -- Correlation coefficient in the range [-1, 1], or None if there is an error.
    """

    # Parameter validation
    if len(args) != 2:
        print("The function expects 2 arguments: (vet1, vet2).")
        return None

    vet1, vet2 = args

    # Check if vet1 and vet2 are vectors (one-dimensional)
    if vet1.ndim != 1 or vet2.ndim != 1:
        print("The parameters 'vet1' and 'vet2' must be one-dimensional vectors.")
        return None

    # Check if vet1 and vet2 have the same length
    if len(vet1) != len(vet2):
        print("The vectors 'vet1' and 'vet2' must have the same length.")
        return None

    # Ensure vet1 and vet2 are column vectors
    vet1 = vet1.reshape(-1)
    vet2 = vet2.reshape(-1)

    # Calculation of covariance and standard deviations
    Sxy = sum((vet1 - mean(vet1)) * (vet2 - mean(vet2)))
    Sx = sqrt(sum((vet1 - mean(vet1)) ** 2))
    Sy = sqrt(sum((vet2 - mean(vet2)) ** 2))

    # Calculation of the correlation coefficient
    if Sx == 0 or Sy == 0:
        print("The standard deviations of 'vet1' or 'vet2' are zero, correlation cannot be calculated.")
        return None

    r = Sxy / (Sx * Sy)

    # Check if r is a number
    if isnan(r):
        print("The correlation coefficient is not a valid number.")
        return None

    return r
#==============================================================================

#==============================================================================
def rmse(*args):
    """
    The rmse function calculates the root mean square error between two vectors.

    Parameters:
    vet1 -- data vector (nx1) (Mandatory)
    vet2 -- data vector (nx1) (Mandatory)

    Returns:
    R -- RMSE value

    Example:
    vet1 = np.linspace(0, 1, 100)
    vet2 = np.random.randn(100)
    R = rmse(x, y)
    """

    if len(args) != 2:
        print("This function requires 2 parameters (Input, Output)")
        return None
    vet1, vet2 = args
    # Check if inputs are vectors and have the same size
    if not isinstance(vet1, ndarray) or not isinstance(vet2, ndarray):
        print("Inputs must be numpy arrays.")
        return None
    if len(vet1) != len(vet2):
        print("Input vectors must have the same size.")
        return None

    # Ensure vet1 and vet2 are vectors (1D)
    if vet1.ndim == 1:
        vet1 = array([vet1]).T
    if vet2.ndim == 1:
        vet2 = array([vet2]).T

    if (vet1.ndim != 2 and vet1.shape[1] != 1) or (vet2.ndim != 2 and vet2.shape[1] != 1):
        print("Input signals must be one-dimensional vectors.")
        return None

    # Calculate RMSE
    R = sqrt(sum((vet1 - vet2) ** 2) / len(vet1))

    return R
#==============================================================================

#==============================================================================
def aic(*args):
    """
    Computes the AIC (Akaike Information Criterion) as:
    AIC = log(r) + 2 * n

    Parameters via args:
    1. n (int)      -- number of parameters in the model
    2. r (numpy array) -- variance of the residuals

    Returns:
    f (float) -- calculated AIC value

    """

    # Argument validation
    if len(args) != 2:
        print("The function expects 2 arguments: (n, r)")
        return None

    n, r = args

    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        print("The parameter 'n' must be a positive integer.")
        return None

    # Check if r is a float
    if not isinstance(r, float):
        print("The parameter 'r' must be a float.")
        return None

    # AIC calculation
    f = 2 * n - 2 * log(r)
    return f
#==============================================================================

#==============================================================================
def likelihood(y_true, y_pred):
    """
    Calculates the likelihood function for a linear regression model.

    Parameters:
    y_true (array): Observed values (real data).
    y_pred (array): Predicted values by the model.

    Returns:
    float: Likelihood value (log-likelihood).
    """
    n = len(y_true)
    residuals = y_true - y_pred
    #obtaining the standard deviation of the errors
    sigma = std(residuals) / 5
    log_likelihood = -n / 2 * log(2 * pi * sigma**2) - sum(residuals**2) / (2 * sigma**2)
    return log_likelihood
#==============================================================================

#==============================================================================
def akaike(*args):
    """
    Calculates the Akaike Information Criterion (AIC) to select the number of 
    process terms in polynomial models.

    Parameters (via *args):
        1. model -- matrix code of candidate terms (numpy array)
        2. u     -- system input data (numpy array)
        3. y     -- system output data (numpy array)

    Returns:
        f    -- sequence containing the Akaike criterion data
        mint -- the number of process terms to be considered
    """

    # Parameter validation
    if len(args) != 3:
        print("The function expects 3 arguments: (model, u, y).")
        return None, None

    model, u, y = args

    # Check if parameters are numpy arrays
    if not isinstance(model, ndarray) or not isinstance(u, ndarray) or not isinstance(y, ndarray):
        print("The parameters 'model', 'u', and 'y' must be numpy arrays.")
        return None, None

    # Check if model is two-dimensional and u, y are one-dimensional
    if model.ndim != 2:
        print("The parameter 'model' must be a matrix (2D array).")
        return None

    if u.ndim != 1 or y.ndim != 1:
        print("The parameters 'u' and 'y' must be one-dimensional vectors.")
        return None

    model_pr = model
    l, c = model_pr.shape
    yhat = zeros((len(y), 0))
    F = zeros((l, 1))

    #-------------------------------------------------------------------------------
    for i in range(l):

        Parameters, model_ols = ols(model, u, y, i+1)
        yhat = simodeld(model_ols, Parameters, u, 0)
        y_arr = array([y]).T
        x = likelihood(y_arr, yhat)
        F[i] = aic(i+1, abs(x))

    Nter = argmin(array(abs(F)))
    return F, Nter
#==============================================================================

#==============================================================================
def funcorr(*args):
    """
    Calculates the autocorrelation function of the variable x.

    Parameters:
    x -- data vector (numpy array) (Mandatory)
    max_des -- maximum lag to be analyzed (int) (Mandatory)
    grafico -- true or false (Mandatory)

    Returns:
    H -- Vector of lags used for the function calculation
    Y -- Autocorrelation function of x
    r -- correlation coefficient
    """
    if len(args) != 3:
        print("This function requires 2 mandatory parameters: x (numpy array), and maximum lag (int)")
        return None, None
    x, max_des, grafico = args

    # Parameter validation
    if not isinstance(x, ndarray) or x.ndim != 2:
        print("The parameter 'x' must be a one-dimensional vector (numpy array).")
        return None, None, None

    if not isinstance(max_des, (int, integer)) or max_des < 0:
        print('The variable "max_des" must be a non-negative integer.')
        return None, None, None

    # Initialize lists for lags and correlations
    H = []
    Y = []

    # Calculate the correlation function
    if max_des >= len(x):
        max_des = len(x) - 1

    for h in range(max_des + 1):
        H.append(h)
        # Shift x by h elements
        xh = roll(x, h)
        # Calculate the correlation coefficient
        Sxy = sum((x - mean(x)) * (xh - mean(xh)))
        Sx = sqrt(sum((x - mean(x)) ** 2))
        Sy = sqrt(sum((xh - mean(xh)) ** 2))

        if Sx == 0 or Sy == 0:
            Y.append(None)  # Returns NaN if the standard deviation is zero
        else:
            r = Sxy / (Sx * Sy)
            Y.append(r)
    Y = array(Y)
    H = array(H)
    if grafico:
        plot_funcorr(H, Y)
    r = mean(r)
    return H, Y, r
#==============================================================================

#==============================================================================
def plot_funcorr(H, Y):
    """
    Plots the autocorrelation function.

    Parameters:
    H -- Vector of lags.
    Y -- Autocorrelation function.
    """
    figure()
    plot(H, Y, label='Correlation Function')
    axhline(0.08, color='red', linestyle='--', label='Uncertainty threshold of +0.08')
    axhline(-0.08, color='green', linestyle='--', label='Uncertainty threshold of -0.08')
    grid()
    xlabel('H (lag)')
    ylabel('f(H)')
    title('Correlation Function')
    legend()
    show()
#==============================================================================

#==============================================================================
def funcorrcruz(*args):
    """
    Calculates the cross-correlation function between variables x and y.

    Parameters:
    x -- data vector (numpy array) (Mandatory)
    y -- data vector (numpy array) (Mandatory)
    max_des -- maximum lag to be analyzed (int) (Mandatory)

    Returns:
    H -- Vector of lags used for the function calculation
    Y -- Cross-correlation function between x and y
    """
    if len(args) != 3:
        print("This function requires 3 mandatory parameters: x (numpy array), y (numpy array), and maximum lag (int)")
        return None, None

    x, y, max_des = args
    # Parameter validation
    if not isinstance(x, ndarray) or not isinstance(y, ndarray):
        print("The parameters 'x' and 'y' must be vectors (numpy arrays).")
        return None, None

    if x.ndim != 1 or y.ndim != 1:
        print("The parameters 'x' and 'y' must be one-dimensional vectors.")
        return None, None

    if len(x) != len(y):
        print("The signals 'x' and 'y' must have the same dimension.")
        return None, None

    if not isinstance(max_des, (int, integer)) or max_des < 0:
        print('The variable "max_des" must be a non-negative integer.')
        return None, None

    # Initialize lists for lags and correlations
    H = []
    Y = []

    # Calculate the cross-correlation function
    if max_des >= len(x):
        max_des = len(x) - 1

    for h in range(max_des + 1):
        H.append(h)
        # Shift y by h elements
        yh = roll(y, h)
        # Calculate the correlation coefficient
        Sxy = sum((x - mean(x)) * (yh - mean(yh)))
        Sx = sqrt(sum((x - mean(x)) ** 2))
        Sy = sqrt(sum((yh - mean(yh)) ** 2))

        if Sx == 0 or Sy == 0:
            Y.append(None)  # Returns NaN if the standard deviation is zero
        else:
            Y.append(Sxy / (Sx * Sy))

    Y = array(Y)
    H = array(H)
    plot_funcorrcruz(H, Y)
    return H, Y
#===============================================================================

#===============================================================================
def plot_funcorrcruz(H, Y):
    """
    Plots the cross-correlation function.

    Parameters:
    H -- Vector of lags.
    Y -- Cross-correlation function.
    """
    figure()
    plot(H, Y, label='Cross-Correlation Function')
    axhline(0.08, color='red', linestyle='--', label='Uncertainty threshold of +0.08')
    axhline(-0.08, color='green', linestyle='--', label='Uncertainty threshold of -0.08')
    grid()
    xlabel('H (lag)')
    ylabel('f(H)')
    title('Cross-Correlation Function')
    legend()
    show()
#===============================================================================

#===============================================================================
def narx(**kwargs):
    """
    NARX function for modeling nonlinear systems.

    Parameters:
    -----------
    u : array-like
        Input vector of the system, representing the control or excitation input of the model. Should be passed as a list or array of size N (total number of samples).

    y : array-like
        Observed output vector of the system, representing the response data to be modeled. Should be passed as a list or array of size N (same size as u).

    time : array-like
        Time vector corresponding to each sample of u and y. Should be passed as a list or array of size N.

    gnl : int
        Degree of nonlinearity to be considered in the model.

    lagy : int
        Number of lags of the output (y) to be considered in the modeling.

    lagu : int
        Number of lags of the input (u) to be considered in the modeling.

    lage : int
        Number of lags of the error to be considered in the modeling.

    y0 : float
        Initial value of the system output used in the model validation simulation.

    max_lags_coor : int
        Maximum number of lags for error signal correlation during model validation.
    cluster: int array (n,2)
        Clusters to be removed from the model
    chart : bool
        If True, the function will generate graphs to visually validate the simulation compared to the real data.
    """

    # Validation of mandatory inputs
    required_keys = ["u", "y", "time", "gnl", "lagy", "lagu", "y0", "max_lags_coor", "chart"]

    # Check if all mandatory keys are present
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Error: The argument '{key}' is mandatory.")
            return

    # Convert data to NumPy arrays
    u = kwargs["u"]
    y = kwargs["y"]
    time = kwargs["time"]
    cluster = None
    valid_cluster = False
    # Validate if the input vector is a NumPy array
    if not isinstance(u, ndarray) and len(u) > 1:
        u = array(u)
    # Validate if the output vector is a NumPy array
    if not isinstance(y, ndarray) and len(y) > 1:
        y = array(y)
    # Validate if the time vector is a NumPy array
    if not isinstance(time, ndarray) and len(time) > 1:
        time = array(time)
    # Consistency checks for vector sizes
    if len(u) != len(y) or len(u) != len(time):
        raise ValueError("Error: The vectors 'u', 'y', and 'time' must have the same size.")
        return

    # Validation of data types for other parameters
    if not isinstance(kwargs["gnl"], int) or kwargs["gnl"] <= 0:
        raise ValueError("Error: The parameter 'gnl' must be a positive integer.")
        return

    if not isinstance(kwargs["lagy"], int) or kwargs["lagy"] < 0:
        raise ValueError("Error: The parameter 'lagy' must be a non-negative integer.")
        return

    if not isinstance(kwargs["lagu"], int) or kwargs["lagu"] < 0:
        raise ValueError("Error: The parameter 'lagu' must be a non-negative integer.")
        return

    if not isinstance(kwargs["y0"], (int, float)):
        raise ValueError("Error: The parameter 'y0' must be a number (int or float).")
        return

    if not isinstance(kwargs["max_lags_coor"], int) or kwargs["max_lags_coor"] <= 0:
        raise ValueError("Error: The parameter 'max_lags_coor' must be a positive integer.")
        return

    if not isinstance(kwargs["chart"], bool):
        raise ValueError("Error: The parameter 'chart' must be a boolean value (True or False).")
        return
    #cluster Validation
    if 'cluster' in kwargs:
        cluster = kwargs["cluster"]
        if not isinstance(cluster, ndarray) and len(u) > 1:
            cluster = array(cluster)
        if cluster.shape[1] == 2:
            valid_cluster = True
        if not valid_cluster:
            print("Cluster invalid, the data provided will not be used on the estimator")     
            cluster = None
    # Continue with the function execution if all validations are passed
    gnl = kwargs["gnl"]
    lagy = kwargs["lagy"]
    lagu = kwargs["lagu"]
    y0 = kwargs["y0"]
    max_lags_coor = kwargs["max_lags_coor"]
    chart = kwargs["chart"]

    # Start data separation
    length = len(u)
    center = length // 2
    ui = u[:center]
    uv = u[center:]
    yi = y[:center]
    yv = y[center:]
    ti = time[:center]
    tv = time[center:]
    
    #==============================================================================
    # Generate the set of candidate terms and get necessary information
    #==============================================================================
    model, Tter = genterms(gnl, lagy, lagu)

    #==============================================================================
    # Cluster's removing
    #==============================================================================
    if valid_cluster:
      for i in range(len(cluster)):
          print(f"{cluster[i,0]},{cluster[i,1]}")
          model = mcand(model,cluster[i,0],cluster[i,1])

    model_pr, model_no, ntp, ntr, gnl, nu, ny, ne, mlag = get_info(model)

    #==============================================================================
    # Using criteria for selecting candidate terms
    #==============================================================================
    model, variance, residuals = sort_pr(model_pr, ui, yi)

    F, ntr = akaike(model, ui, yi)

    #==============================================================================
    # Parameter estimation using the conventional least squares method
    #==============================================================================
    Parameters, model = ols(model, ui, yi, ntr)

    #==============================================================================
    # Simulate the model for a validation input
    #==============================================================================
    yhat = simodeld(model, Parameters, uv, y0)

    #==============================================================================
    # Model validation through the error signal correlation function of the simulation
    #==============================================================================
    result = array([yv]).T - yhat
    H, Y, r = funcorr(result, max_lags_coor, chart)

    if chart:
        #==============================================================================
        # Visual validation
        #==============================================================================
        figure(figsize=(10, 6))
        plot(tv, yv, color="blue", label='Real Data')
        plot(tv, yhat, color="red", label='Simulated Output')
        title("Simulated Output vs Real Data")
        xlabel("Time (ms)")
        ylabel("Voltage (V)")
        legend()
        grid()
        show()

    #==============================================================================
    # RMSE Value
    #==============================================================================
    R = rmse(yhat, yv)
    ysim = yhat

    return model, Parameters, uv, ui, yv, yi, tv, ti, ysim, R, r, F
#==============================================================================


#===============================================================================
def narmax(**kwargs):
    """
    NARMAX function for modeling nonlinear systems.

    Parameters:
    -----------
    u : array-like
        Input vector of the system, representing the control or excitation input of the model. Should be passed as a list or array of size N (total number of samples).

    y : array-like
        Observed output vector of the system, representing the response data to be modeled. Should be passed as a list or array of size N (same size as u).

    time : array-like
        Time vector corresponding to each sample of u and y. Should be passed as a list or array of size N.

    gnl : int
        Degree of nonlinearity to be considered in the model.

    lagy : int
        Number of lags of the output (y) to be considered in the modeling.

    lagu : int
        Number of lags of the input (u) to be considered in the modeling.

    lage : int
        Number of lags of the error to be considered in the modeling.

    y0 : float
        Initial value of the system output used in the model validation simulation.

    max_lags_coor : int
        Maximum number of lags for error signal correlation during model validation.

    nir: int
        Number of noise interations

    cluster: int array (n,2)
        Clusters to be removed from the model

    chart : bool
        If True, the function will generate graphs to visually validate the simulation compared to the real data.
    """

    # Validation of mandatory inputs
    required_keys = ["u", "y", "time", "gnl", "lagy", "lagu", "lage", "y0", "max_lags_coor", "chart", "nir" , "ntr"]

    # Check if all mandatory keys are present
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Error: The argument '{key}' is mandatory.")
            return

    # Convert data to NumPy arrays
    u = kwargs["u"]
    y = kwargs["y"]
    time = kwargs["time"]
    cluster = None
    valid_cluster = False
    # Validate if the input vector is a NumPy array
    if not isinstance(u, ndarray) and len(u) > 1:
        u = array(u)
    # Validate if the output vector is a NumPy array
    if not isinstance(y, ndarray) and len(y) > 1:
        y = array(y)
    # Validate if the time vector is a NumPy array
    if not isinstance(time, ndarray) and len(time) > 1:
        time = array(time)
    # Consistency checks for vector sizes
    if len(u) != len(y) or len(u) != len(time):
        raise ValueError("Error: The vectors 'u', 'y', and 'time' must have the same size.")
        return

    # Validation of data types for other parameters
    if not isinstance(kwargs["gnl"], int) or kwargs["gnl"] <= 0:
        raise ValueError("Error: The parameter 'gnl' must be a positive integer.")
        return

    if not isinstance(kwargs["lagy"], int) or kwargs["lagy"] < 0:
        raise ValueError("Error: The parameter 'lagy' must be a non-negative integer.")
        return

    if not isinstance(kwargs["lagu"], int) or kwargs["lagu"] < 0:
        raise ValueError("Error: The parameter 'lagu' must be a non-negative integer.")
        return
    if not isinstance(kwargs["lage"], int) or kwargs["lage"] < 0:
        raise ValueError("Error: The parameter 'lagu' must be a non-negative integer.")
        return    

    if not isinstance(kwargs["y0"], (int, float)):
        raise ValueError("Error: The parameter 'y0' must be a number (int or float).")
        return

    if not isinstance(kwargs["max_lags_coor"], int) or kwargs["max_lags_coor"] <= 0:
        raise ValueError("Error: The parameter 'max_lags_coor' must be a positive integer.")
        return

    if not isinstance(kwargs["nir"], int) or kwargs["nir"] <= 0:
        raise ValueError("Error: The parameter 'nir' must be a positive integer.")
        return
    if not isinstance(kwargs["ntr"], int) or kwargs["ntr"] <= 0:
        raise ValueError("Error: The parameter 'nir' must be a positive integer.")
        return

    if not isinstance(kwargs["chart"], bool):
        raise ValueError("Error: The parameter 'chart' must be a boolean value (True or False).")
        return
    #cluster Validation
    if 'cluster' in kwargs:
        cluster = kwargs["cluster"]
        if not isinstance(cluster, ndarray) and len(u) > 1:
            cluster = array(cluster)
        if cluster.shape[1] == 2:
            valid_cluster = True
        if not valid_cluster:
            print("Cluster invalid, the data provided will not be used on the estimator")     
            cluster = None
    # Continue with the function execution if all validations are passed
    gnl = kwargs["gnl"]
    lagy = kwargs["lagy"]
    lagu = kwargs["lagu"]
    lage = kwargs["lage"]
    y0 = kwargs["y0"]
    max_lags_coor = kwargs["max_lags_coor"]
    nir = kwargs["nir"]
    ntr = kwargs["ntr"]
    chart = kwargs["chart"]

    # Start data separation
    length = len(u)
    center = length // 2
    ui = u[:center]
    uv = u[center:]
    yi = y[:center]
    yv = y[center:]
    ti = time[:center]
    tv = time[center:]
    
    #==============================================================================
    # Generate the set of candidate terms and get necessary information
    #==============================================================================
    model, Tter = genterms(gnl, lagy, lagu,lage)

    #==============================================================================
    # Cluster's removing
    #==============================================================================
    if valid_cluster:
      for i in range(len(cluster)):
          print(f"{cluster[i,0]},{cluster[i,1]}")
          model = mcand(model,cluster[i,0],cluster[i,1])

    model_pr, model_no, ntp, ntr_model, gnl, nu, ny, ne, mlag = get_info(model)

    #==============================================================================
    # Using criteria for selecting candidate terms
    #==============================================================================
    model, variancia, residuos = sort_pr(model_pr, ui, yi)
   
    F, ntp = akaike(model, ui, yi)

    #==============================================================================
    # Parameter estimation using the Extended Least Square
    #==============================================================================
    Parameters, Variance, model_pr, model_no, ParProcess, ParNoise, M_Parameters = els(model, ntp, ntr, nir,ui, yi )

    #==============================================================================
    # Simulate the model for a validation input
    #==============================================================================
    yhat = simodeld(model_pr, ParProcess, uv, y0)

    #==============================================================================
    # Model validation through the error signal correlation function of the simulation
    #==============================================================================
    result = array([yv]).T - yhat
    H, Y, r = funcorr(result, max_lags_coor, chart)

    if chart:
        #==============================================================================
        # Visual validation
        #==============================================================================
        figure(figsize=(10, 6))
        plot(tv, yv, color="blue", label='Real Data')
        plot(tv, yhat, color="red", label='Simulated Output')
        title("Simulated Output vs Real Data")
        xlabel("Time (ms)")
        ylabel("Voltage (V)")
        legend()
        grid()
        show()

    #==============================================================================
    # RMSE Value
    #==============================================================================
    R = rmse(yhat, yv)
    ysim = yhat

    return model, ParProcess, uv, ui, yv, yi, tv, ti, ysim, R, r, F
#==============================================================================