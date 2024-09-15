<<<<<<< HEAD
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from hierarchical_loss import HierarchicalLoss, HierarchicalLossConvex
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
import math
import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == '__main__':
    
    download_cifar = True  # SET TO TRUE THE FIRST TIME
    use_hierarchical, use_convex, LEARNING_RATE, n_epochs,desired_classes,parent_list = sys.argv[1]=='True', sys.argv[2]=='True', float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
    str_list = desired_classes.strip("[]")
    desired_classes = str_list.replace("'", "").split(", ")
    str_list2 = parent_list.strip("[]")
    parent_list = str_list2.split(", ")
    print('The desired classes and the hierarchical structure are ', desired_classes, parent_list)
    for i in range(len(parent_list)) :
        parent_list[i] = eval(parent_list[i])

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    writer = SummaryWriter(log_dir=f"models/run_{timestamp}_{'h_loss' if use_hierarchical else 'ce_loss'}_{'convex' if use_convex else 'not_convex'}")

    torch.manual_seed(1)
    torch.autograd.set_detect_anomaly(True)

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    #data processing 

    transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                    transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                    transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                    transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                            ])
    batch_size = 16

    all_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = tuple(desired_classes)
    #classes = ('car', 'cat', 'dog', 'horse', 'truck')
    num_classes = len(desired_classes)

    train = True

    train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=download_cifar, transform=transform_train)
    filtered_indices_train = [idx for idx, (_, label) in enumerate(train_dataset_full) if train_dataset_full.classes[label] in classes]

    train_dataset = torch.utils.data.Subset(train_dataset_full, filtered_indices_train)

    print(f"the number of desired classes is {num_classes}" )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=download_cifar, transform=transform)
    # idx for idx, (_, label) in enumerate(test_dataset_full) if test_dataset_full.classes[label] in classes]
    filtered_indices_test = [idx for idx, (_, label) in enumerate(test_dataset_full) if test_dataset_full.classes[label] in classes]
    test_dataset = torch.utils.data.Subset(test_dataset_full, filtered_indices_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Adding some images on Tensorboard

    examples = iter(test_loader)
    example_data, example_targets = next(examples)
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('our image' ,img_grid)

    # Definig the model 

    class LeNet(nn.Module): # https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch
        def __init__(self, nb_classes = num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
            self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # We double the feature maps for every conv layer as in pratice it is really good.
            self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.fc1 = nn.Linear(4*4*64, 500) # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(500, 100) # output nodes are 10 because our dataset have 10 different categories
            self.fc3 = nn.Linear(100, nb_classes) # output nodes are 10 because our dataset have 10 different categories
        def forward(self, x):
            x = F.relu(self.conv1(x)) #Apply relu to each output of conv layer.
            x = F.max_pool2d(x, 2, 2) # Max pooling layer with kernal of 2 and stride of 2
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*64) # flatten our images to 1D to input it to the fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout1(x) # Applying dropout b/t layers which exchange highest parameters. This is a good practice
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    def filtered_to_decimal_label(label):
        transformation = []
        for elem in classes:

            transformation.append(all_classes.index(elem))
        
        transformed_label = [transformation[l] for l in label]
        return transformed_label

    def decimal_to_filtered_label(label):
        transformation = []
        for elem in all_classes:
            transformation.append(classes.index(elem) if elem in classes else None)
        transformed_label = [transformation[l] for l in label]
        return torch.tensor(transformed_label, dtype=torch.int64)

    def evaluate_model(net, criterion, epoch):
        print('---------------MODEL EVALUATION---------------')
        for loader, dataset_name in zip([train_loader, test_loader], ['train', 'test']):
            correct = 0
            total = 0
            print(classes)
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            running_loss=0.0
            with torch.no_grad():
                for data in tqdm(loader, total=len(loader)):
                    images, labels = data
                    labels = decimal_to_filtered_label(labels)
                    outputs = net(images)
                    if use_hierarchical: 
                        outputs = criterion.convert_logits_to_class_preds(outputs.data)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    running_loss +=loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            if label=='car':
                                print(prediction)
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
               

            avg_loss = running_loss / len(loader)
            accuracy = 100 * correct / total
            
            writer.add_scalar(f'validation_loss/{dataset_name}', avg_loss, epoch)
            print(f'Accuracy of the network on the {len(loader)*batch_size} {dataset_name} images: {100 * correct // total} %')
            writer.add_scalar(f'global_validation_accuracy/{dataset_name}', 100*correct // total, epoch)
            accuracies = {}
            for classname, correct_count in correct_pred.items():
                if total_pred[classname] == 0:
                    print(classname)
                else:
                    accuracy = 100 * float(correct_count) / total_pred[classname]
                    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
                    accuracies[classname] = accuracy
            writer.add_scalars(f'class_accuracy/{dataset_name}', accuracies, epoch)
        return 100 * correct / total

    ce_loss = nn.CrossEntropyLoss()
    h_loss = HierarchicalLossConvex(parent_list) if use_convex else HierarchicalLoss(parent_list)
    


    if use_hierarchical:
        criterion = h_loss
    
        net = LeNet(nb_classes=criterion.nb_of_nn_outputs)
    else:
        criterion = ce_loss
        net = LeNet()

    writer.add_graph(net, example_data)
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    with open('trials.txt', 'a') as f:
        f.write('\n' + datetime.now().strftime("%Y_%m_%d_%Hh%M") + '\n')
        f.write(f"{'hierarchical' if use_hierarchical else 'cross-entropy'} loss\n")
        f.write(f"{'convex' if use_convex  else 'not_convex'} loss\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of epochs: {n_epochs}\n")
        f.write('\n')
    
    if train:
        print('---------------MODEL TRAINING---------------')
        with open('trials.txt', 'a') as f:
            f.write("Epoch\tTraining Loss\tTraining Accuracy\n")
            for epoch in range(n_epochs):
                print(f">>>>> Epoch {epoch}")
                running_loss = 0.0
                for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                    inputs, labels = data
                    labels = decimal_to_filtered_label(labels)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                f.write(f"epoch {epoch}\n")
                epoch_accuracy = evaluate_model(net, criterion, epoch)
                f.write(f"{epoch}\t{running_loss/len(train_loader)}\t{epoch_accuracy}\n")
                print("Loss for epoch:", running_loss/len(train_loader))
                writer.add_scalar(
                "loss/training_loss",
                running_loss/len(train_loader),
                epoch,
                )
                writer.add_scalar(
                "global_training_accuracy",
                epoch_accuracy,
                epoch,
                )
                torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "criterion": criterion,
                    "lr": LEARNING_RATE,
                    },
                    f"models/model_{timestamp}_{'h_loss' if use_hierarchical else 'ce_loss'}.pt",
                )

                if math.isnan(running_loss/len(train_loader)):
                    break
        
print('training finished')
=======
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from hierarchical_loss import HierarchicalLoss, HierarchicalLossConvex
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
import math
import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == '__main__':
    
    download_cifar = True  # SET TO TRUE THE FIRST TIME
    use_hierarchical, use_convex, LEARNING_RATE, n_epochs,desired_classes,parent_list = sys.argv[1]=='True', sys.argv[2]=='True', float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
    str_list = desired_classes.strip("[]")
    desired_classes = str_list.replace("'", "").split(", ")
    str_list2 = parent_list.strip("[]")
    parent_list = str_list2.split(", ")
    print('The desired classes and the hierarchical structure are ', desired_classes, parent_list)
    for i in range(len(parent_list)) :
        parent_list[i] = eval(parent_list[i])

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    writer = SummaryWriter(log_dir=f"models/run_{timestamp}_{'h_loss' if use_hierarchical else 'ce_loss'}_{'convex' if use_convex else 'not_convex'}")

    torch.manual_seed(1)
    torch.autograd.set_detect_anomaly(True)

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    #data processing 

    transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                    transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                    transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                    transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                            ])
    batch_size = 16

    all_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = tuple(desired_classes)
    #classes = ('car', 'cat', 'dog', 'horse', 'truck')
    num_classes = len(desired_classes)

    train = True

    train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=download_cifar, transform=transform_train)
    filtered_indices_train = [idx for idx, (_, label) in enumerate(train_dataset_full) if train_dataset_full.classes[label] in classes]

    train_dataset = torch.utils.data.Subset(train_dataset_full, filtered_indices_train)

    print(f"the number of desired classes is {num_classes}" )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=download_cifar, transform=transform)
    # idx for idx, (_, label) in enumerate(test_dataset_full) if test_dataset_full.classes[label] in classes]
    filtered_indices_test = [idx for idx, (_, label) in enumerate(test_dataset_full) if test_dataset_full.classes[label] in classes]
    test_dataset = torch.utils.data.Subset(test_dataset_full, filtered_indices_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Adding some images on Tensorboard

    examples = iter(test_loader)
    example_data, example_targets = next(examples)
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('our image' ,img_grid)

    # Definig the model 

    class LeNet(nn.Module): # https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch
        def __init__(self, nb_classes = num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
            self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # We double the feature maps for every conv layer as in pratice it is really good.
            self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.fc1 = nn.Linear(4*4*64, 500) # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(500, 100) # output nodes are 10 because our dataset have 10 different categories
            self.fc3 = nn.Linear(100, nb_classes) # output nodes are 10 because our dataset have 10 different categories
        def forward(self, x):
            x = F.relu(self.conv1(x)) #Apply relu to each output of conv layer.
            x = F.max_pool2d(x, 2, 2) # Max pooling layer with kernal of 2 and stride of 2
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*64) # flatten our images to 1D to input it to the fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout1(x) # Applying dropout b/t layers which exchange highest parameters. This is a good practice
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    def filtered_to_decimal_label(label):
        transformation = []
        for elem in classes:

            transformation.append(all_classes.index(elem))
        
        transformed_label = [transformation[l] for l in label]
        return transformed_label

    def decimal_to_filtered_label(label):
        transformation = []
        for elem in all_classes:
            transformation.append(classes.index(elem) if elem in classes else None)
        transformed_label = [transformation[l] for l in label]
        return torch.tensor(transformed_label, dtype=torch.int64)

    def evaluate_model(net, criterion, epoch):
        print('---------------MODEL EVALUATION---------------')
        for loader, dataset_name in zip([train_loader, test_loader], ['train', 'test']):
            correct = 0
            total = 0
            print(classes)
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            running_loss=0.0
            with torch.no_grad():
                for data in tqdm(loader, total=len(loader)):
                    images, labels = data
                    labels = decimal_to_filtered_label(labels)
                    outputs = net(images)
                    if use_hierarchical: 
                        outputs = criterion.convert_logits_to_class_preds(outputs.data)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    running_loss +=loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            if label=='car':
                                print(prediction)
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
               

            avg_loss = running_loss / len(loader)
            accuracy = 100 * correct / total
            
            writer.add_scalar(f'validation_loss/{dataset_name}', avg_loss, epoch)
            print(f'Accuracy of the network on the {len(loader)*batch_size} {dataset_name} images: {100 * correct // total} %')
            writer.add_scalar(f'global_validation_accuracy/{dataset_name}', 100*correct // total, epoch)
            accuracies = {}
            for classname, correct_count in correct_pred.items():
                if total_pred[classname] == 0:
                    print(classname)
                else:
                    accuracy = 100 * float(correct_count) / total_pred[classname]
                    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
                    accuracies[classname] = accuracy
            writer.add_scalars(f'class_accuracy/{dataset_name}', accuracies, epoch)
        return 100 * correct / total

    ce_loss = nn.CrossEntropyLoss()
    h_loss = HierarchicalLossConvex(parent_list) if use_convex else HierarchicalLoss(parent_list)
    


    if use_hierarchical:
        criterion = h_loss
    
        net = LeNet(nb_classes=criterion.nb_of_nn_outputs)
    else:
        criterion = ce_loss
        net = LeNet()

    writer.add_graph(net, example_data)
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    with open('trials.txt', 'a') as f:
        f.write('\n' + datetime.now().strftime("%Y_%m_%d_%Hh%M") + '\n')
        f.write(f"{'hierarchical' if use_hierarchical else 'cross-entropy'} loss\n")
        f.write(f"{'convex' if use_convex  else 'not_convex'} loss\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of epochs: {n_epochs}\n")
        f.write('\n')
    
    if train:
        print('---------------MODEL TRAINING---------------')
        with open('trials.txt', 'a') as f:
            f.write("Epoch\tTraining Loss\tTraining Accuracy\n")
            for epoch in range(n_epochs):
                print(f">>>>> Epoch {epoch}")
                running_loss = 0.0
                for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                    inputs, labels = data
                    labels = decimal_to_filtered_label(labels)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                f.write(f"epoch {epoch}\n")
                epoch_accuracy = evaluate_model(net, criterion, epoch)
                f.write(f"{epoch}\t{running_loss/len(train_loader)}\t{epoch_accuracy}\n")
                print("Loss for epoch:", running_loss/len(train_loader))
                writer.add_scalar(
                "loss/training_loss",
                running_loss/len(train_loader),
                epoch,
                )
                writer.add_scalar(
                "global_training_accuracy",
                epoch_accuracy,
                epoch,
                )
                torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "criterion": criterion,
                    "lr": LEARNING_RATE,
                    },
                    f"models/model_{timestamp}_{'h_loss' if use_hierarchical else 'ce_loss'}.pt",
                )

                if math.isnan(running_loss/len(train_loader)):
                    break
        
print('training finished')
>>>>>>> 508682ee0d514b5ff061619de687e210a9c13e63
