'''
PYTHONPATH=$(pwd) python scripts/merge_data.py 
'''


import pickle



num_thread = 10

imdb = []
for i in range(num_thread):
    imdb += pickle.load(open(f'/home/ubuntu/visualDet3D/visualDet3D/workspace/Mono3D_nuscenes_450x800/output/validation/imdb_{i}.pkl', 'rb'))


pickle.dump(imdb, open('/home/ubuntu/visualDet3D/visualDet3D/workspace/Mono3D_nuscenes_450x800/output/validation/imdb.pkl', 'wb'))


imdb = []
for i in range(num_thread):
    imdb += pickle.load(open(f'/home/ubuntu/visualDet3D/visualDet3D/workspace/Mono3D_nuscenes_450x800/output/training/imdb_{i}.pkl', 'rb'))


pickle.dump(imdb, open('/home/ubuntu/visualDet3D/visualDet3D/workspace/Mono3D_nuscenes_450x800/output/training/imdb.pkl', 'wb'))

