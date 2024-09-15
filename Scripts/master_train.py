import subprocess
from datetime import datetime

trials=[
# Here we test the default cross-entropy loss function for the image classification model 
  {
        'use_hierarchical':'False',
        'use_convex':'False',
        'learning_rate':'0.001',
        'nbr_epochs':'60',
        'classes':"['ship', 'airplane', 'cat', 'dog', 'automobile', 'truck']",
        'parent_list': "[None, 0, 0, 1, 1, 1, 2, 2, 3, 3]",
    },

# Here we test the non-convex hierarchical loss function for the image classification model 
    {
        'use_hierarchical':'True',
        'use_convex':'False',
        'learning_rate':'0.001',
        'nbr_epochs':'60',
        'classes':"['ship', 'airplane', 'cat', 'dog', 'automobile', 'truck']",
        'parent_list': "[None, 0, 0, 1, 1, 1, 2, 2, 3, 3]",
    },
# Here we test the convex hierarchical loss function for the image classification model 
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
    subprocess.call(['python', 'cifar.py', trial['use_hierarchical'], trial['use_convex'], trial['learning_rate'], trial['nbr_epochs'],trial['classes'], trial['parent_list']])