import numpy as np

def multi_regress(y,Z):
    """Performs multiple linear regression.
        
        Parameters
        ----------
        y : array_like, shape = (n,) or (n,1)
        The vector of dependent variable data
        
        Z : array_like, shape = (n,m)
        The matrix of independent variable data
        
        Returns
        -------
        numpy.ndarray, shape = (m,) or (m,1)
        The vector of model coefficients
        
        numpy.ndarray, shape = (n,) or (n,1)
        The vector of residuals
        
        float
        The coefficient of determination, r^2

        numpy.ndarray, shape = (n,) or (n,1)
        The vector of model outputs
        """
    #setup matrices to find coefficients 
    ZTZ = np.matmul(np.transpose(Z),Z)
    ZTy = np.matmul(np.transpose(Z),y)
    a = np.matmul(np.linalg.inv(ZTZ),ZTy)

    #calculate mean model
    T_model_avg = np.full(y.shape,np.mean(y))
    e_avg = y - T_model_avg
    S_t = np.sum(np.square(e_avg))

    #calculate first order linear model
    T_model_linear = np.matmul(Z,a)
    e_linear = y - T_model_linear
    S_r = np.sum(np.square(e_linear))

    rsq = (S_t - S_r) / S_t
    
    e = e_linear
    
    return a, e, rsq, T_model_linear