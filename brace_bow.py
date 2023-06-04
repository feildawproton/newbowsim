import numpy as np
np.set_printoptions(linewidth=np.inf)
import matplotlib.pyplot as plt
from typing import Tuple

from build_bow import build_unbraced_limb
from Elements.limb_element import T_g2l_limb
from BowModel.Dimensions import Draw
from assembly import calc_T_g2l_limb, Ms_l2g, insert_gKlimb, update_pos

print("RECALCULATE FORCE IN CASE THAT IT GOES DOWN!!!!!")  

#pull direction does not need to be normalized, this function will handle that
#incrementing force will happen outside this function
#it could possibly go negative
def p_driven(gK: np.ndarray, pull_node: int, pull_vec: np.ndarray):
    #fix the brace point by assigning it's force as unknown
    #and it's displacement as known

    #check that the pull node makes sense
    #it cannot be at the brace node
    #and it cannot be past the len of the dofs
    #also ckeck that the pull vector as len 2
    assert pull_node != 0
    assert pull_node * 3 + 2 < gK.shape[0]
    assert pull_vec.size == 2

    #set the pull porce
    #p will be 0 everywhere except where it is pulled from
    #note that p[pull_node * 3 + 2] is implied to be zero (no rotational external forces)
    #even the first node gets assigned [0, 0, 0] even though we do not use it
    #it's force as unknown and we account for that by using a submatrix of gK and sampling of p
    #it's displacement as known [0, 0, 0] and we will account for that by appending [0, 0, 0] and the u_sub vector
    #we form a full size p here so that we don't make indexing with pull_node easier to think about
    assert gK.shape[-2] == gK.shape[-1]
    p                       = np.zeros(gK.shape[-1])
    
    p[pull_node * 3]        = pull_vec[0]  #u component (x direction)
    p[pull_node * 3 + 1]    = pull_vec[1]  #v component (y direction)

    #now we need to solve for the unknown dispacements
    #all forces are known except the first node
    #all displacements are unknown except the first node
    #recall that the displacement in the rist node are [0, 0, 0]
    p_sub   = p[3:]
    gK_sub  = gK[3:,3:]
    u_sub   = np.linalg.solve(gK_sub, p_sub)
    u       = np.append(np.array([0,0,0]), u_sub)
    
    return u

def brace_bow(X_0: np.ndarray, Y_0: np.ndarray, Kp_limb: np.ndarray):
    #begin by setting X to X_0 and Y to Y_=0
    #use np.copy() instead of assignment because it is more explicit
    X = np.copy(X_0)
    Y = np.copy(Y_0)

    #confirm the number of nodes and elements make sense
    #in the case of just the limb there should be 1 more than number of elements
    num_nodes = X_0.size
    num_elems = Kp_limb.shape[0]
    assert num_nodes == Y_0.size == num_elems + 1
    gK_i = np.zeros([3*num_nodes, 3*num_nodes])

    #since we are resetting K every iteration (to account for rotation)
    #we need to collect p as a sum
    p = np.zeros(num_nodes * 3)
    while np.min(Y) > - Draw["Brace height"]:
        #print("iter: ", i)
        #i += 1

        # --REFORM K --
        #reform K outside of the force increment

        #caculate/recalculate the transorms 
        #these transforms are in the form of global to local
        #this is because _Ms_l2g expects them that way and will perform the approptiate transpose or to put them in global
        #then perform the transformation of the list of gK
        
        Ts_limb = calc_T_g2l_limb(X, Y)
        gKs_limb = Ms_l2g(Kp_limb, Ts_limb)
        
        #reform gK
        #as to not recreate gK, zero it out before reforming
        gK_i[:,:] = 0
        insert_gKlimb(gK_i, gKs_limb)
        
        #we will increment on this pull vector until we update X and Y
        #the pull node will always be n_nodes -1 in this formulation
        #thhe pull direction is from the last node (of the limb in this to the nock)
        x_nock      = 0
        y_nock      = - Draw["Brace height"]
        x_node      = X[num_nodes - 1]
        y_node      = Y[num_nodes - 1]
        ref_vec     = np.array([[x_nock - x_node],[y_nock - y_node]])
        norm        = np.linalg.norm(ref_vec)
        ref_vec     = np.divide(ref_vec, norm)

        #we will iterate for until we get the change in u that we wnat
        p_inc   = .0005
        p_itr   = p_inc
        u_i     = np.zeros(3 * num_nodes)
        itrs    = 0
        while u_i[-2] > - 0.5:
            #run the iteratrion
            pull_vec    = np.multiply(p_itr, ref_vec)
            u_i         = p_driven(gK_i, num_nodes - 1, pull_vec)
            p_itr      += p_inc
            itrs       += 1
        print("Sub iterations: ", itrs)

        #once we've met our condition, update X and Y
        X, Y  = update_pos(X, Y, u_i)
        print("min Y now: ", np.min(Y))

        p_i = np.matmul(gK_i, u_i)
        p = np.add(p_i, p)

    return X, Y, p
        

if __name__ == "__main__":
    X_unbraced, Y_unbraced, Mp_limb, Kp_limb, T_limb_unbraced = build_unbraced_limb()
    
    

    X, Y, p = brace_bow(X_unbraced, Y_unbraced, Kp_limb)
    print("resulting p: ")
    print(p)
    print("min y", np.min(Y))
    print("requirement: ", - Draw["Brace height"])
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title("Bracing the bow")
    ax.plot(X_unbraced, Y_unbraced, color = "red")
    ax.plot(X, Y, color = "blue")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    plt.show() 

   