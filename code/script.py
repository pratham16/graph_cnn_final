import os

os.system(' export CUDA_VISIBLE_DEVICES="" ')

K_range = [2,3,4,5,7,9,11,13,15,20]
for K in K_range:
    print('K=',K)
    os.system('python DPP4_graph_conv.py '+str(K)+' >results/output_graph_conv_K='+str(K))
    os.system('python DPP4_graph_conv_fc.py '+str(K)+' >results/output_graph_conv_fc_K='+str(K))
