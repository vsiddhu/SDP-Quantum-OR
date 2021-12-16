#Date: December 15, 2021
#This code contains material supporting a paper, currently titled *Five
#Starter Pieces: Quantum Information Science via Semi-definite Programs*, by
#Vikesh Siddhu (vsiddhu@protonmail.com) and Sridhar Tayur (stayur@cmu.edu). 
#The paper is available on the arXiv(https://arxiv.org/). 
#The arXiv paper is released there is under the arXiv.org perpetual, non-exclusive
#license (https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html), and
#this code is released under the MIT license (https://opensource.org/licenses/MIT).

# In[2]:


import numpy as np
import picos as pic


# In[3]:


def krausToChoiJ(krsLst):
    r"""Takes in list of Kraus operators for the channel and
    returns the Choi-Jamiolkowski representation for the
    channel and its complement
    
    Parameters
    ----------
    krsLst : list
             A list of 2-D numpy arrays
             
    Returns
    ----------
    cjBA : 2-D numpy array
           Choi-Jamiolkowski representation of the channel.
           
    cjCA : 2-D numpy array
           Choi-Jamiolkowski representation of the complementary channel
    """
    #dc,db,da dimensional array
    channelKet = np.array(krsLst)
    channelKetConj = channelKet.conj()
    (dc,db,da) = np.shape(channelKet)
    
    #Choi-Jamiolkowski rep for the direct channel: contract c indices
    cjBA=np.einsum('ijk,ist->jkst',channelKet,channelKetConj)
    cjBA=np.reshape(cjBA,(db*da,db*da))
    
    #Choi-Jamiolkowski rep for the complementary channel: contract b indices
    cjCA=np.einsum('ijk,rjt->ikrt',channelKet,channelKetConj)
    cjCA=np.reshape(cjCA,(dc*da,dc*da))
    
    return(cjBA, cjCA)


# In[4]:


def choiJToLinPic(jbaP,da,db):
    r"""Takes the Choi-Jamiolkowski representation, the
    input and output dimensions for the channel and returns
    the transfer matrix
    
    Parameters
    ----------
    JBbaP : picos.expressions.exp_affine.AffineExpression
           Choi-Jamiolkowski representation of the channel
    
    da : int
         Input dimension of the channel
    
    db : int
         Output dimension of the channel
             
    Returns
    ----------
    TBba : picos.expressions.exp_affine.AffineExpression
           Transfer matrix for the channel, dimension (db*db, da*da)
    """

    TbaDP = jbaP.reshuffled(permutation='ikjl', dimensions=(db,da,db,da),order='C')
    return TbaDP.T.reshaped((da*da,db*db)).T


# In[5]:


def linTochoiJPic(TbaP,da,db):
    r"""Takes the Choi-Jamiolkowski representation, the
    input and output dimensions for the channel and returns
    the transfer matrix
    
    Parameters
    ----------
    TBbaP : picos.expressions.exp_affine.AffineExpression
            Transfer matrix for the channel
    
    da : int
         Input dimension of the channel
    
    db : int
         Output dimension of the channel
             
    Returns
    ----------
    JBba : picos.expressions.exp_affine.AffineExpression
           Choi-Jamiolkowski representation for the channel, dimension (db*da, db*da)
    """
    JbaP = TbaP.reshuffled(permutation='ikjl', dimensions=(db,db,da,da),order='C')
    return JbaP.T.reshaped((da*db, da*db)).T


# In[6]:


def choiJOfChanInSeriesPic(cj1, cj2,da,db,dc):
    r"""Takes as input the Choi-Jamiolkowski representation
    of two channels one and two, and returns the Choi-Jamiolkowski representation
    of channel two applied after channel one in series
    
    Parameters
    ----------
    cj1 : picos.expressions.exp_affine.AffineExpression
          Choi-Jamiolkowski rep of the first channel
    
    cj2 : picos.expressions.exp_affine.AffineExpression
          Choi-Jamiolkowski rep of the second channel
          
    da : int
         input dimension of the first channel
         
    db  : int
         output dimension of the first channel
    
    dc : int
          output dimension of the second channel
          
    Returns
    ----------
    ch21 : 2-d Numpy Array
    
    """
    Tba = choiJToLinPic(cj1,da,db)
    Tcb = choiJToLinPic(cj2,db,dc)
    Tac = Tcb*Tba
    return linTochoiJPic(Tac,da,dc)


# In[27]:


def boundFunc(eps,dc):
    r"""Takes as input an the value epsilon for which a channel
    is epsilon degradable, and the dimension of the channel's
    environment. Returns a value which bounds the difference
    between the channel's quantum capacity its coherent information
    
    Parameters
    ----------
    eps : float
    
    dc : int
         dimension of the channel's environment/Choi-Rank
         
    Returns
    ----------
    bnd : float
          value bounding the difference between a channel's
          capacity and its coherent information
    """
    if dc <= 1:
        print("Environment dimension must be greater than one, returning zero")
        return 0
    ent = lambda x : -x*np.log2(x) - (1-x)*np.log2(1-x)
    
    bnd = eps*np.log2(dc-1)/2
    bnd += eps*np.log2(dc)
    bnd += ent(eps/2)
    bnd += (1 + eps/2)*ent(eps/(2+eps))
    
    return bnd

