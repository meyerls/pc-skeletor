from pc_skeletor import LBC
pcd='/home/se86kimy/Dropbox/07_data/TreeSkeleton/tree_skeleton_01-03/03/tree_03.ply'
lbc = LBC(point_cloud=pcd,
          down_sample=0.01,
          init_contraction=2)
lbc.extract_skeleton()
lbc.extract_topology()
lbc.visualize()
#lbc.show_graph(lbc.skeleton_graph)
#lbc.show_graph(lbc.topology_graph)
lbc.export_results('/home/se86kimy/Dropbox/07_data/TreeSkeleton/tree_skeleton_01-03/03/output')
