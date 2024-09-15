import torch.nn as nn
import torch
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

def get_path_from_root(idx, parent_indexes):
    if idx !=0: # if not root
        return get_path_from_root(parent_indexes[idx], parent_indexes) + [idx] 
    return []

def get_children(target_parent_idx, parent_indexes):
    '''Get the list of the children of target_parent_idx'''
    return [index for index, parent_idx in enumerate(parent_indexes) if parent_idx == target_parent_idx]

def get_children_list_deep(target_parent_idx, children_list):
    '''Get the list of all the children of target_parent_idx and their children until the bottom of the tree'''
    children_list_deep = children_list[target_parent_idx].copy()
    for child in children_list[target_parent_idx]:
        children_list_deep += get_children_list_deep(child, children_list)
    return children_list_deep

def get_nb_nn_outputs(children_list:list[int]):
    nb_of_nn_outputs = 0
    for node in children_list:
        if len(node) == 2:
            nb_of_nn_outputs +=1 # sigmoid will be used because there are only 2 children 
        elif len(node) >= 3:
            nb_of_nn_outputs += len(node)
    return nb_of_nn_outputs

kl_loss = nn.KLDivLoss(reduction='sum')
class HierarchicalLoss(nn.Module):
    '''Implementation of the hierarchical loss. The input is parent_list, which contains the index of the parent for each element. 
    parent_list = [None, 0, 0, 1, 1] means that the root (index 0) has 2 children (indexes 1 and 2) and that element number 1 has 2 children (indexes 3 and 4)'''
    def __init__(self, parent_list:list[int]):
        super(HierarchicalLoss, self).__init__()
        self.parent_list = parent_list
        self.nb_h_classes = len(parent_list)
        self.childrens_list = [get_children(idx, parent_list) for idx in range(len(parent_list))]
        self.childrens_list_deep = [get_children_list_deep(idx, self.childrens_list) for idx in range(len(parent_list))]
        self.paths_from_root = [get_path_from_root(idx, parent_list) for idx in range(len(parent_list))]
        self.nb_of_nn_outputs = get_nb_nn_outputs(self.childrens_list)
        self.leafs = self.get_leafs()

    # Initial syntax

    '''def get_probas_from_nn_outputs(self, nn_outputs):
        
        if nn_outputs.size(0) != self.nb_of_nn_outputs:
            raise ValueError("The number of outputs from the model is incorrect to compute the loss given the hierarchical configuration you provided.")
        probas = torch.zeros((self.nb_h_classes, self.nb_h_classes), dtype=nn_outputs.dtype, device=nn_outputs.device)
        nn_outputs_current = nn_outputs.clone()
        for parent_idx, children in enumerate(self.childrens_list):
            if children:
                nb_children = len(children)
                if nb_children == 2:
                    u = nn_outputs_current[0] 
                    u = torch.exp(u)
                    p = u / (1 + u)  
                    probas[parent_idx][children[0]] = p
                    probas[parent_idx][children[1]] = 1 - p
                    nn_outputs_current = nn_outputs_current[1:] 
                if nb_children >= 3:
                    u_list = nn_outputs_current[:nb_children] 
                    u_list = torch.exp(u_list)
                    probas[parent_idx][children] = u_list / torch.sum(u_list)
                    nn_outputs_current = nn_outputs_current[nb_children:] 
        return torch.clamp(probas, 1e-9, 1 - 1e-9)'''
        
    # New optimized syntax  

    def get_probas_from_nn_outputs(self, nn_outputs):
        if nn_outputs.size(0) != self.nb_of_nn_outputs:
            raise ValueError("The number of outputs from the model is incorrect to compute the loss given the hierarchical configuration you provided.")
        
        probas = torch.zeros((self.nb_h_classes, self.nb_h_classes), dtype=nn_outputs.dtype, device=nn_outputs.device)
        start_idx = 0 # We do not clone the outputs of the mdel every time this function is called but we use a start index
        for parent_idx, children in enumerate(self.childrens_list):
            if children:
                nb_children = len(children)
                if nb_children == 2:
                    u = torch.exp(nn_outputs[start_idx])
                    p = u / (1 + u)  
                    probas[parent_idx, children[0]] = p
                    probas[parent_idx, children[1]] = 1 - p
                    start_idx += 1  
                elif nb_children >= 3:
                    u_list = torch.exp(nn_outputs[start_idx:start_idx + nb_children])
                    probas[parent_idx, children] = u_list / u_list.sum()
                    start_idx += nb_children  
        
        return torch.clamp(probas, 1e-9, 1 - 1e-9)
    def get_leafs(self):
        # Returns the list of indexes of the elements which do not have children (leafs of the tree). They correspond to the real classes on the Cifar 10 dataset
        leafs_idx = []
        for i, child in enumerate(self.childrens_list):
            if len(child) == 0:
                leafs_idx.append(i)
        return leafs_idx
    
    def get_binary_vector_label(self, label:int):
        #Converts the label (int) to a binary hierarchical representation marking the class and parent classes.
        leaf_idx = self.leafs[label] # index in hierarchical representation of the leaf
        binary_vector_labels = torch.zeros(self.nb_h_classes, dtype=torch.int64)  
        current_idx = leaf_idx # start from the leaf
        while current_idx != 0: # stop at the root
            binary_vector_labels[current_idx] = 1 # mark the parent class
            current_idx = self.parent_list[current_idx] # go up to the parent
        return binary_vector_labels
    # Initial syntax

    """def convert_logits_to_class_preds(self, logits):
        #Convert nn outputs (logits) to class probabilities (leafs space - no parent classes)
        batch_size = logits.size(0)
        batch_preds = []
        for i in range(batch_size):
            probas = self.get_probas_from_nn_outputs(logits[i]) # probabilities for each class including parent classes
            preds = []
            for elem in self.leafs: # for each class in the initial formulation, really important here we work with the desired classes
                product = 1
                for ancestor_idx in self.paths_from_root[elem]:
                    product *= probas[self.parent_list[ancestor_idx]][ancestor_idx] 
                preds.append(product)
            batch_preds.append(torch.tensor(preds))
        return torch.stack(batch_preds)"""
    
    # Optimzed syntax

    def convert_logits_to_class_preds(self, logits):
        #Convert nn outputs (logits) to class probabilities (leafs space - no parent classes)
        batch_size = logits.size(0)
        batch_preds = []
        for i in range(batch_size):
            probas = self.get_probas_from_nn_outputs(logits[i]) # probabilities for each class including parent classes
            preds = []
            for elem in self.leafs: # for each class in the initial formulation, really important here we work with the desired classes
                product = torch.prod(torch.tensor(
                    [probas[self.parent_list[ancestor_idx]][ancestor_idx] for ancestor_idx in self.paths_from_root[elem]], # line changed to do the product with torch.prod !!!!
                    ))
                preds.append(product)
            batch_preds.append(torch.tensor(preds))
        return torch.stack(batch_preds)
    
    def draw_graph(self, classes_name = None):
        '''Draws a graph representation of the class tree.'''
        G = nx.DiGraph() # create graph object
        for child_idx, parent_idx in enumerate(self.parent_list): # add nodes
            G.add_node(child_idx)
            if parent_idx is not None:
                G.add_edge(parent_idx, child_idx)
        pos = graphviz_layout(G, prog="dot") # draw with a tree layout
        plt.figure(figsize=(8, 6))
        if classes_name:
            labels={}
            for i, leaf_idx in enumerate(self.leafs):
                labels[leaf_idx] = classes_name[i]
            nx.draw(G, pos, with_labels=False, node_size=500, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True, arrowstyle='->')
            nx.draw_networkx_labels(G, pos, labels, font_size=8, verticalalignment="bottom")
        else:
            nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True, arrowstyle='->')
        plt.title("Arbre à partir de la liste parent-enfant")
        plt.show()

    # Initial Syntax
        
    '''def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        loss_per_sample = torch.zeros(batch_size)
        for i in range(batch_size):
            sample_input = inputs[i]
            sample_target = targets[i]
            probas = self.get_probas_from_nn_outputs(sample_input)
            binary_label = self.get_binary_vector_label(sample_target) 
            indexes_to_sum = [torch.tensor([idx] + self.childrens_list_deep[idx]) for idx in range(self.nb_h_classes)] 
            descendant_label_sum = torch.zeros(self.nb_h_classes)
            for j in range(self.nb_h_classes):
                descendant_label_sum[j] = torch.sum(binary_label[indexes_to_sum[j]])
            loss = torch.zeros(1, dtype=inputs.dtype, device=inputs.device)
            for idx in range(1, self.nb_h_classes):
                loss -= descendant_label_sum[idx] * torch.log(probas[self.parent_list[idx]][idx])
                if len(self.childrens_list[idx]) > 0: 
                    product = 1
                    for ancestor_idx in self.paths_from_root[idx]:
                        product *= probas[self.parent_list[ancestor_idx]][ancestor_idx]
                    loss += product 
            loss_per_sample[i] = loss
        return loss_per_sample.mean()'''
    
    #Optimized syntax

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        loss_per_sample = torch.zeros(batch_size)
        
        for i in range(batch_size):
            sample_input = inputs[i]
            sample_target = targets[i]
            probas = self.get_probas_from_nn_outputs(sample_input)
            binary_label = self.get_binary_vector_label(sample_target) 
            indexes_to_sum = [torch.tensor([idx] + self.childrens_list_deep[idx]) for idx in range(self.nb_h_classes)] 
            descendant_label_sum = [torch.sum(binary_label[indexes_to_sum[j]]) for j in range(self.nb_h_classes)]
            loss = torch.zeros(1, dtype=inputs.dtype, device=inputs.device)
            for idx in range(1, self.nb_h_classes):
                loss -= descendant_label_sum[idx] * torch.log(probas[self.parent_list[idx]][idx])
                # change in the method used to calculate the product, here we use torch.prod which optimizes this process 
                if self.childrens_list[idx]:
                    product = torch.prod(torch.tensor(
                    [probas[self.parent_list[ancestor_idx]][ancestor_idx] for ancestor_idx in self.paths_from_root[idx]],
                    ))
                    loss += product
            loss_per_sample[i] = loss
        
        return loss_per_sample.mean()
class HierarchicalLossConvex(nn.Module):
    '''Implementation of the hierarchical loss in the convex version. The input is parent_list, which contains the index of the parent for each element. 
    parent_list = [None, 0, 0, 1, 1] means that the root (index 0) has 2 children (indexes 1 and 2) and that element number 1 has 2 children (indexes 3 and 4)'''
    def __init__(self, parent_list:list[int]):
        super(HierarchicalLossConvex, self).__init__()
        self.parent_list = parent_list
        self.nb_h_classes = len(parent_list)
        self.childrens_list = [get_children(idx, parent_list) for idx in range(len(parent_list))]
        self.childrens_list_deep = [get_children_list_deep(idx, self.childrens_list) for idx in range(len(parent_list))]
        self.paths_from_root = [get_path_from_root(idx, parent_list) for idx in range(len(parent_list))]
        self.nb_of_nn_outputs = get_nb_nn_outputs(self.childrens_list)
        self.leafs = self.get_leafs()
        self.depths = [len(path)+1 for path in self.paths_from_root]

    def get_probas_from_nn_outputs(self, nn_outputs):
        if nn_outputs.size(0) != self.nb_of_nn_outputs:
            raise ValueError("The number of outputs from the model is incorrect to compute the loss given the hierarchical configuration you provided.")
        
        probas = torch.zeros((self.nb_h_classes, self.nb_h_classes), dtype=nn_outputs.dtype, device=nn_outputs.device)
        start_idx = 0 # change: we no longer clone the outputs of the model but define a starting index
        for parent_idx, children in enumerate(self.childrens_list):
            if children:
                nb_children = len(children)
                if nb_children == 2:
                    u = torch.exp(nn_outputs[start_idx])
                    p = u / (1 + u)  
                    probas[parent_idx, children[0]] = p
                    probas[parent_idx, children[1]] = 1 - p
                    start_idx += 1  
                elif nb_children >= 3:
                    u_list = torch.exp(nn_outputs[start_idx:start_idx + nb_children])
                    probas[parent_idx, children] = u_list / u_list.sum()
                    start_idx += nb_children  
        
        return torch.clamp(probas, 1e-9, 1 - 1e-9)
    #turn into a comprehension list 
    def get_leafs(self):
        '''Returns the list of indexes of the elements which do not have children (leafs of the tree). They correspond to the subclasses.'''
        leafs_idx = []
        for i, child in enumerate(self.childrens_list):
            if len(child) == 0:
                leafs_idx.append(i)
        return leafs_idx
    
    def get_binary_vector_label(self, label:int):
        '''Converts the label (int) to a binary hierarchical representation marking the class and parent classes.'''
        leaf_idx = self.leafs[label] # index in hierarchical representation of the leaf
        binary_vector_labels = torch.zeros(self.nb_h_classes, dtype=torch.int64)  
        current_idx = leaf_idx # start from the leaf
        while current_idx != 0: # stop at the root
            binary_vector_labels[current_idx] = 1 # mark the parent class
            current_idx = self.parent_list[current_idx] # go up to the parent
        return binary_vector_labels
    
    def convert_logits_to_class_preds(self, logits):
        '''Convert nn outputs (logits) to class probabilities (leafs space - no parent classes)'''
        batch_size = logits.size(0)
        batch_preds = []
        for i in range(batch_size):
            probas = self.get_probas_from_nn_outputs(logits[i]) # probabilities for each class including parent classes
            preds = []
            for elem in self.leafs: # for each class in the initial formulation
                product = 1
                
                product = torch.prod(torch.tensor(
                    [probas[self.parent_list[ancestor_idx]][ancestor_idx] for ancestor_idx in self.paths_from_root[elem]],
                    ))
                preds.append(product)
            batch_preds.append(torch.tensor(preds))        
        return torch.stack(batch_preds)
    


    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        loss_per_sample = torch.zeros(batch_size)
        for i in range(batch_size):
            probas = self.get_probas_from_nn_outputs(inputs[i])
            binary_label = self.get_binary_vector_label(targets[i]).float()
            loss = torch.zeros(1, dtype=inputs.dtype, device=inputs.device)
            for parent_idx, children in enumerate(self.childrens_list):
                if children:
                    p = probas[parent_idx][children]
                    q = binary_label[children]
                    loss += (1/self.depths[parent_idx]) * kl_loss(torch.log(p), q) # Utilizing the hierarchical structure, we penalize more the errors made on superclasses  
                    #loss += kl_loss(torch.log(p), q)
            loss_per_sample[i] = loss
        return loss_per_sample.mean()



