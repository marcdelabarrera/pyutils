# Standard libraries
from warnings import warn

# Third party libraries
from scipy.optimize import root_scalar, RootResults, root
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

def plot_roots(fun:callable, bracket:list[float], num:int=50):
    x = np.linspace(*bracket,num)
    fig,ax=plt.subplots()
    ax.plot(x, [fun(i) for i in x])
    ax.axhline(0, color='black')
    plt.show()


def count_roots(fun:callable, bracket:list[float], num:int=50)->int:
    '''
      Count the number of roots of a function in a given interval
      Parameters
      ----------
      fun: callable
      bracket:
    '''
    x = np.linspace(*bracket,num)
    y = np.array([fun(i) for i in x])
    sign_y = y>0
    return np.sum(np.diff(sign_y))


def brackets_with_sign_change(x:ndarray,y:ndarray)->list[tuple[float,float]]:
  '''
  Given a function f(x) and a set of points x, find the brackets where the function changes sign
  Parameters
  ----------
  x: Array
  y: Array
  Examples
  --------
  >>> x = np.arange(-10,10,0.1)
  >>> y = np.sin(x)
  find_brackets(x,y)
  TODO
  '''
  return [(i,j) for i,j in zip(x[:-1][y[:-1]*y[1:]<0],x[1:][y[:-1]*y[1:]<0])]


def rroot_scalar(fun: callable, bracket: list[float], num:int=100, warn_multiple_sol:bool=True)->list[RootResults]:
  '''
  robust root scalar. Finds all roots of an scalar function in a given interval.
  Parameters
  ----------
  f: callable
  '''
  x = np.linspace(*bracket,num)
  brackets = brackets_with_sign_change(x, fun(x))
  if len(brackets)==0:
      raise ValueError('No sign change found in the entire bracket')
  
  solutions = [root_scalar(fun, bracket = bracket) for bracket in brackets]
  
  if len(solutions)>1 and warn_multiple_sol:
       warn(f'Multiple solutions found: {solutions}')
  return solutions


def rroot(fun, x0, tol=1e-6, has_aux = False, **kwargs)->RootResults:
    '''
    Robust root scalar that checks if the solution is correct.
    Parameters
    ----------
    fun: callable
    x0: float
    tol:  float, optional
    has_aux: bool, optional
        Indicates whether fun returns more than one value. The mathematical
        output is the first element. Default False.
    '''
    if has_aux:
      solution = root(lambda x: fun(x)[0], x0 = x0, **kwargs)
    else:
      solution = root(fun, x0 = x0, **kwargs)
    #if not solution.success:
    #  raise ValueError(f'No solution {fun} with x0 = {x0}\n{solution}')
    if np.max(np.abs(solution.fun))>tol:
      raise ValueError(f'No solution {fun} with x0 = {x0}\n{solution}')
    return solution



