## Usage de ce code

Ce code nous a permis de faire l'optimisation du code des fonctions de perte hiérarchiques convexe et non convexe. Tandis que la comparaison de performance entres les différentes fonctions loss utilisés. 

## Exécution du code 

### Résultats de la comparaison des trois fonctions de perte : 

Il suffit de mettre les trois fichier master_train.py, cifar.py et hierarchical_loss.py dans le même dossier et d'exécuter master_train.py, Il faut allez ensuite à cifar.py et ouvrir Tensorboard pour visualiser les pertes des fonctions loss et leur précisions globales et sur les diverses classes. 

### Résultats de l'optimisation : 

Pour répliquer la méthode utilisée pour l'optimisation du code. Il faut d'abord installer scalene avec pip, puis dans le terminal et en étant dans le dossier ou les trois programmes python sont stockés écrire la commande suivante  scalene --html --outfile optimization.html cifar.py True False 0.01 1 "['automobile', 'cat', 'dog', 'horse', 'truck']" "[None, 0, 0, 1, 2, 2, 2, 1]". 
Les deux pages donnant les résultats d'optimization sont données et sont nommées  Temps_de_calcul_avant_opti.html Temps_de_calcul_après_opti.html. Le résultat à partir de CProfile (profiling_result.txt) est aussi donné pour expliquer les opérations intensives que la fonction de perte hiérarchique implémente, mais sans figurer dans le code. 



