#Date: December 15, 2021
#This code contains material supporting a paper, currently titled *Five
#Starter Pieces: Quantum Information Science via Semi-definite Programs*, by
#Vikesh Siddhu (vsiddhu@protonmail.com) and Sridhar Tayur (stayur@cmu.edu). 
#The paper is available on the arXiv(https://arxiv.org/). 
#The arXiv paper is released there is under the arXiv.org perpetual, non-exclusive
#license (https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html), and
#this code is released under the MIT license (https://opensource.org/licenses/MIT).


import numpy as np
import picos as pic


# In[5]:


def swapLastTwoOfThreePartyState(rhoaB,da,db):
    r"""Takes a tri-partite density operator performs
    swap on the last two spaces and returns the result
    
    Parameters
    ----------
    rhoaBP : picos.expressions.exp_affine.AffineExpression
             tri-partite density operator
    
    da : int
         Dimension of first space
    
    db : int
         Dimension of each of the two second spaces
             
    Returns
    ----------
    SrhoaB : picos.expressions.exp_affine.AffineExpression
             tri-partite density operator
    """
    d = da*db*db
    
    SrhoaB = rhoaB.reshuffled(permutation='ikjlnm', dimensions=(da,db,db,da,db,db),order='C')
    SrhoaB = SrhoaB.T.reshaped((d,d)).T
    return(SrhoaB)

