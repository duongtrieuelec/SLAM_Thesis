import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    s -- scaling factor
    
    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    model_norm = np.linalg.norm(model_zerocentered)
    data_norm = np.linalg.norm(data_zerocentered)
    print("Model norm: %f, Data norm: %f"%(model_norm, data_norm))
    
    W = np.zeros((3,3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.identity(3)
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    
    # Estimate optimal scale
    s = 1.0
    rot_model = rot * model_zerocentered
    dots = 0.0
    norms = 0.0
    for column in range(model.shape[1]):
        dots += np.dot(data_zerocentered[:,column].transpose(), rot_model[:,column])
        norms += np.dot(model_zerocentered[:,column].transpose(), model_zerocentered[:,column])
    print("Align debug: dots=%f, norms=%f"%(dots, norms))
    
    # Use bounding-box diagonal ratio for scale estimation.
    # This directly captures the "visual size" of each trajectory.
    model_min = np.min(model, 1)
    model_max = np.max(model, 1)
    model_diag = np.linalg.norm(model_max - model_min)
    
    data_min = np.min(data, 1)
    data_max = np.max(data, 1)
    data_diag = np.linalg.norm(data_max - data_min)
    
    s = data_diag / model_diag
    print("Using Bounding-Box Diagonal Scale: %f"%s)
    
    trans = data.mean(1) - s * np.dot(rot, model.mean(1))
    
    model_aligned = s * rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error,s

def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = np.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[0][i])
            y.append(traj[1][i])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)

if __name__=="__main__":
    # parse command line
    import argparse
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the aligned trajectories to an image file (format: png or jpg)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args()

    first_list = {}
    second_list = {}
    
    # Read files
    for f, d in [(args.first_file, first_list), (args.second_file, second_list)]:
        for line in open(f).readlines():
            if line.startswith("#"): continue
            tokens = line.split()
            if len(tokens) < 8: continue
            stamp = float(tokens[0])
            tx = float(tokens[1])
            ty = float(tokens[2])
            tz = float(tokens[3])
            d[stamp] = [tx, ty, tz]
    
    matches = associate(first_list, second_list, float(args.offset), float(args.max_difference))    
    if len(matches) < 2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

    first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
    second_xyz = np.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
    
    rot,trans,trans_error,s = align(second_xyz,first_xyz)
    print("estimated_scale %f"%s)
    
    second_xyz_aligned = s * rot * second_xyz + trans
    
    first_stamps = list(first_list.keys())
    first_stamps.sort()
    first_xyz_full = np.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
    
    second_stamps = list(second_list.keys())
    second_stamps.sort()
    second_xyz_full = np.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = s * rot * second_xyz_full + trans
    
    if args.verbose:
        print("compared_pose_pairs %d pairs"%(len(trans_error)))

        print("absolute_translational_error.rmse %f m"%np.sqrt(np.dot(trans_error,trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m"%np.mean(trans_error))
        print("absolute_translational_error.median %f m"%np.median(trans_error))
        print("absolute_translational_error.std %f m"%np.std(trans_error))
        print("absolute_translational_error.min %f m"%np.min(trans_error))
        print("absolute_translational_error.max %f m"%np.max(trans_error))
    else:
        print(np.sqrt(np.dot(trans_error,trans_error) / len(trans_error)))
        
    if args.save_associations:
        file = open(args.save_associations,"w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f"%(a,x1,y1,z1,b,x2,y2,z2) for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A)]))
        file.close()
        
    if args.save:
        file = open(args.save,"w")
        file.write("\n".join(["%f %f %f %f"%(b,x2,y2,z2) for (a,b),x2,y2,z2 in zip(matches,second_xyz_full_aligned.transpose().A[:,0].tolist(),second_xyz_full_aligned.transpose().A[:,1].tolist(),second_xyz_full_aligned.transpose().A[:,2].tolist())]))
        file.close()

    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_traj(ax,first_stamps,first_xyz_full.transpose().A.transpose(),'-','black',"ground truth")
        plot_traj(ax,second_stamps,second_xyz_full_aligned.transpose().A.transpose(),'-','blue',"estimated")
        
        # Plot ground truth and estimated trajectories
        # label="difference"
        # for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A):
        #     ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
        #     label=""
            
        ax.legend()
        ax.axis('equal') 
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.savefig(args.plot,format="png")
