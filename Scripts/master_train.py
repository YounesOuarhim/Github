import subprocess
from datetime import datetime

trials=[
    
    {
        'use_hierarchical':'False',
        'use_convex':'False',
        'learning_rate':'0.001',
        'nbr_epochs':'60',
        'classes':"['ship', 'airplane', 'cat', 'dog', 'automobile', 'truck']",
        'parent_list': "[None, 0, 0, 1, 1, 1, 2, 2, 3, 3]",
    },
    {
        'use_hierarchical':'True',
        'use_convex':'False',
        'learning_rate':'0.001',
        'nbr_epochs':'60',
        'classes':"['ship', 'airplane', 'cat', 'dog', 'automobile', 'truck']",
        'parent_list': "[None, 0, 0, 1, 1, 1, 2, 2, 3, 3]",
    },
    {
        'use_hierarchical':'True',
        'use_convex':'True',
        'learning_rate':'0.001',
        'nbr_epochs':'60',
        'classes':"['ship', 'airplane', 'cat', 'dog', 'automobile', 'truck']",
        'parent_list': "[None, 0, 0, 1, 1, 1, 2, 2, 3, 3]",
    },
]

for trial in trials:
    subprocess.call(['python', 'C:\\Users\\dell\\OneDrive\\Bureau\\segmentation\\segmentation\\loss_alix\\hierarchical\cifar.py', trial['use_hierarchical'], trial['use_convex'], trial['learning_rate'], trial['nbr_epochs'],trial['classes'], trial['parent_list']])