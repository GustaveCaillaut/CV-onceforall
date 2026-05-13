# Mini-OFA pour CIFAR-10 — README détaillé

Ce dépôt contient une implémentation compacte et modifiable d’un pipeline inspiré de **Once-for-All (OFA)** pour CIFAR-10. Le fichier principal est :

```text
ofa_cifar10_options.py
```

Le fixed_v2 ne contient aucune option d'optimisation, mais son comportement peut être reproduit avec les options, donc _options.py devrait suffire

L’objectif est de reproduire, à une échelle raisonnable, les grandes étapes d’OFA :

1. entraîner un **supernet** qui partage ses poids entre de nombreux sous-réseaux ;
2. utiliser un entraînement progressif de type **progressive shrinking** ;
3. échantillonner des sous-réseaux et mesurer leur accuracy ;
4. entraîner un **MLP accuracy predictor** ;
5. rechercher une architecture sous contrainte de coût, ici des **MACs** ;
6. évaluer proprement le meilleur sous-réseau trouvé.

Ce code n’est pas une reproduction exacte du code officiel OFA/ImageNet. Il est volontairement plus simple, écrit pour CIFAR-10, et conçu pour expérimenter facilement sur les stratégies de sampling.

---

## 1. Vue d’ensemble du pipeline

Le pipeline complet est :

```text
Étape 1 : train_supernet
    Entraîne un supernet OFA-like sur CIFAR-10.

Étape 2 : sample_acc_dataset
    Tire N sous-réseaux, calibre leurs BatchNorm, mesure leur accuracy,
    et écrit un dataset JSONL du type architecture -> accuracy.

Étape 3 : train_acc_predictor
    Entraîne un MLP qui prédit l’accuracy depuis l’encodage de l’architecture.

Étape 4 : search
    Recherche évolutionnaire sous contrainte MACs avec le predictor.

Étape 5 : eval_subnet
    Évalue réellement le sous-réseau trouvé.
```

---

## 2. Installation

### Dépendances Python

Le script utilise :

```text
torch
torchvision
matplotlib
```

Installation typique :

```bash
pip install torch torchvision matplotlib
```

Pour une carte NVIDIA, vérifier que PyTorch voit CUDA :

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Le script télécharge CIFAR-10 automatiquement via `torchvision.datasets.CIFAR10(..., download=True)`.

---

## 3. Structure des sorties

Si `--save-dir ./runs/ofa_cifar10`, le script produit typiquement :

```text
runs/ofa_cifar10/
├── run_args.json
├── acc_dataset_1000.jsonl
├── accuracy_predictor.pt
├── checkpoints/
│   ├── supernet_best.pt
│   ├── supernet_last.pt
│   └── supernet_epoch_XXX.pt
└── search/
    ├── best_spec.json
    └── search_history.json
```

### Fichiers importants

#### `supernet_best.pt`

Checkpoint qui maximise l’accuracy du **largest subnet** sur la validation pendant l’entraînement.

À utiliser surtout pour tester le plus grand réseau.

#### `supernet_last.pt`

Checkpoint à la fin du progressive shrinking.

À utiliser de préférence pour évaluer les sous-réseaux quelconques, car c’est le supernet final qui a vu tout l’espace élastique.

#### `acc_dataset_*.jsonl`

Dataset d’entraînement du predictor. Une ligne par sous-réseau :

```json
{"spec": {...}, "acc": 74.2, "loss": 0.91, "macs": 81234567}
```

#### `accuracy_predictor.pt`

MLP entraîné pour approximer :

```text
architecture spec -> accuracy mesurée
```

#### `best_spec.json`

Sous-réseau trouvé par la recherche évolutionnaire.

---

## 4. La notion de `spec`

Un sous-réseau est entièrement décrit par un dictionnaire `spec` :

```python
spec = {
    "r": 32,
    "d": [4, 3, 2, 4, 3],
    "ks": [7, 5, 3, 7, ..., 5],  # 20 valeurs
    "e":  [6, 4, 3, 6, ..., 4],  # 20 valeurs
}
```

### Champs

#### `r`

Résolution d’entrée.

Valeurs possibles dans le code :

```python
(24, 28, 32)
```

#### `d`

Profondeur par stage. Il y a 5 stages, donc 5 valeurs.

Valeurs possibles :

```python
(2, 3, 4)
```

Si `d[s] = 2`, alors dans le stage `s`, seuls les deux premiers blocs sont actifs. Les blocs restants sont ignorés.

#### `ks`

Kernel size de chaque bloc élastique. Il y a 5 stages × 4 blocs = 20 blocs, donc 20 valeurs.

Valeurs possibles :

```python
(3, 5, 7)
```

Les petits kernels sont extraits par crop central du plus grand kernel 7×7.

#### `e`

Expansion ratio de chaque bloc élastique. Il y a aussi 20 valeurs.

Valeurs possibles :

```python
(3, 4, 6)
```

Dans cette implémentation, `e` est le paramètre qui joue le rôle de largeur dynamique : plus `e` est grand, plus il y a de canaux intermédiaires dans le bloc MBConv.

---

## 5. Règles d’extraction d’un sous-réseau

Le supernet contient tous les sous-réseaux par partage de poids.

L’extraction est déterministe :

```text
résolution r      -> resize de l’image d’entrée
profondeur d      -> garder les premiers blocs de chaque stage
kernel ks         -> garder le crop central du kernel 7x7
expansion e       -> garder les premiers canaux intermédiaires
```

Cela impose une structure imbriquée :

```text
petit sous-réseau ⊂ grand sous-réseau
```

C’est la raison pour laquelle les sous-réseaux sont représentés par des tailles, pas par des graphes arbitraires.

---

## 6. BatchNorm et calibration BN

### Pourquoi recalibrer les BatchNorm ?

BatchNorm stocke des statistiques :

```text
running_mean
running_var
```

Ces statistiques dépendent de la distribution des activations. Dans OFA, différents sous-réseaux ont des activations différentes. Les statistiques BN apprises pendant l’entraînement du supernet ne sont donc pas toujours adaptées à chaque sous-réseau.

Avant d’évaluer un sous-réseau, on fait donc une **calibration BN** :

```text
1. activer le sous-réseau spec ;
2. reset running_mean/running_var ;
3. passer quelques batches sans gradient ;
4. mettre à jour running_mean/running_var ;
5. évaluer en mode eval().
```

### Détail important du momentum

La fonction `calibrate_bn` utilise un momentum dynamique :

```python
current_momentum = 1.0 / (i + 1)
```

Cela simule une moyenne cumulative. C’est important car après reset :

```text
running_mean = 0
running_var = 1
```

Ces valeurs initiales n’ont pas de signification statistique pour le subnet. Avec une moyenne cumulative, le premier batch remplace réellement l’initialisation, puis les batches suivants construisent une moyenne empirique.

---

## 7. Utilisation complète

### 7.1 Étape 1 — Entraîner le supernet

Commande courte pour debug :

```bat
python ofa_cifar10_options.py train_supernet ^
  --data-dir ./data ^
  --save-dir ./runs/ofa_cifar10 ^
  --epochs-largest 5 ^
  --epochs-kernel 2 ^
  --epochs-depth1 1 ^
  --epochs-depth2 2 ^
  --epochs-width1 1 ^
  --epochs-width2 2 ^
  --channel-sort ^
  --batch-size 128 ^
  --num-workers 0
```

Commande plus sérieuse :

```bat
python ofa_cifar10_options.py train_supernet ^
  --data-dir ./data ^
  --save-dir ./runs/ofa_cifar10 ^
  --epochs-largest 50 ^
  --epochs-kernel 25 ^
  --epochs-depth1 10 ^
  --epochs-depth2 25 ^
  --epochs-width1 10 ^
  --epochs-width2 25 ^
  --channel-sort ^
  --batch-size 128 ^
  --num-workers 0
```

### Paramètres importants

#### `--epochs-largest`

Nombre d’epochs pour entraîner uniquement le plus grand réseau.

#### `--epochs-kernel`

Phase où les kernel sizes deviennent élastiques.

#### `--epochs-depth1`, `--epochs-depth2`

Phases où la profondeur devient élastique.

#### `--epochs-width1`, `--epochs-width2`

Phases où l’expansion ratio devient élastique.

#### `--channel-sort`

Applique un tri des canaux intermédiaires par norme L1 avant les phases width. Cela rend la règle “garder les premiers canaux” plus pertinente.

#### `--shrink-lr`

Learning rate utilisé après la phase largest. Par défaut :

```text
0.005
```

L’idée est que le shrinking est un fine-tuning. Un LR trop élevé peut détruire les poids déjà appris.

---

### 7.2 Étape 2 — Générer le dataset architecture/accuracy

Exemple recommandé :

```bat
python ofa_cifar10_options.py sample_acc_dataset ^
  --data-dir ./data ^
  --save-dir ./runs/ofa_cifar10 ^
  --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_last.pt ^
  --num-arch-samples 1000 ^
  --output ./runs/ofa_cifar10/acc_dataset_1000.jsonl ^
  --bn-calib-batches 5 ^
  --val-batches 10 ^
  --batch-size 512 ^
  --num-workers 0 ^
  --eval-preset accurate
```

### Pourquoi `supernet_last.pt` ?

Le predictor doit apprendre l’accuracy de sous-réseaux quelconques. Il vaut donc mieux utiliser le supernet final, entraîné sur tout l’espace.

### Paramètres importants

#### `--num-arch-samples`

Nombre d’architectures échantillonnées. C’est l’analogue réduit des 16K sous-réseaux du papier OFA.

Expériences intéressantes :

```text
100 / 300 / 500 / 1000 / 2000
```

#### `--bn-calib-batches`

Nombre de batches utilisés pour recalibrer les BN de chaque sous-réseau.

Plus grand = mesure plus fiable mais plus lente.

#### `--val-batches`

Nombre de batches validation utilisés pour estimer l’accuracy.

Si `None`, toute la validation est utilisée.

#### `--sampling-strategy`

Stratégie de sampling :

```text
uniform
small_biased
large_biased
```

---

### 7.3 Étape 3 — Entraîner le predictor

```bat
python ofa_cifar10_options.py train_acc_predictor ^
  --dataset ./runs/ofa_cifar10/acc_dataset_1000.jsonl ^
  --save-dir ./runs/ofa_cifar10 ^
  --predictor-epochs 200 ^
  --predictor-batch-size 64 ^
  --predictor-lr 0.001
```

Le predictor prédit une accuracy en **points de pourcentage**.

Donc :

```text
RMSE = 0.77
```

signifie environ :

```text
erreur typique ≈ 0.77 point d’accuracy
```

Ce n’est pas une erreur relative.

---

### 7.4 Étape 4 — Recherche évolutionnaire

```bat
python ofa_cifar10_options.py search ^
  --predictor ./runs/ofa_cifar10/accuracy_predictor.pt ^
  --save-dir ./runs/ofa_cifar10 ^
  --macs-limit 80000000
```

La recherche maximise :

```text
accuracy prédite
```

sous contrainte :

```text
MACs <= macs-limit
```

Elle produit :

```text
./runs/ofa_cifar10/search/best_spec.json
```

### Tester plusieurs budgets

Pour construire une courbe accuracy/MACs :

```text
40000000
60000000
80000000
100000000
120000000
```

---

### 7.5 Étape 5 — Évaluer réellement le meilleur sous-réseau

Évaluation rapide :

```bat
python ofa_cifar10_options.py eval_subnet ^
  --data-dir ./data ^
  --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_last.pt ^
  --spec-json ./runs/ofa_cifar10/search/best_spec.json ^
  --bn-calib-batches 5 ^
  --test-batches 10 ^
  --batch-size 512 ^
  --num-workers 0 ^
  --eval-preset accurate
```

Évaluation plus propre :

```bat
python ofa_cifar10_options.py eval_subnet ^
  --data-dir ./data ^
  --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_last.pt ^
  --spec-json ./runs/ofa_cifar10/search/best_spec.json ^
  --bn-calib-batches 20 ^
  --batch-size 512 ^
  --num-workers 0 ^
  --eval-preset accurate
```

---

## 8. Les presets d’évaluation

Le code contient :

```python
EVAL_OPTION_PRESETS = {...}
```

Ces presets contrôlent la manière dont les subnets sont évalués.

### `no_opti`

Désactive les optimisations ajoutées.

```python
{
  "cudnn_benchmark": False,
  "cache_calib_on_gpu": False,
  "cache_eval_on_gpu": False,
  "pre_resize_cache": False,
  "bn_calib_loader": "train",
  "restore_bn_after_eval": False,
  "timing": False
}
```

Utilité : reproduire un comportement proche de l’ancien code.

### `accurate`

Preset utilisé pour des mesures relativement fiables, mais avec accélération par cache.

```python
{
  "cudnn_benchmark": True,
  "cache_calib_on_gpu": True,
  "cache_eval_on_gpu": True,
  "pre_resize_cache": True,
  "bn_calib_loader": "train",
  "restore_bn_after_eval": False,
  "timing": True
}
```

Remarque : dans ce code, `accurate` utilise quand même des caches GPU. Le niveau de précision dépend surtout de `--bn-calib-batches` et `--val-batches` / `--test-batches`.

### `balanced`

Comme `accurate`, mais utilise le loader validation pour la calibration BN.

```python
"bn_calib_loader": "val"
```

Souvent plus stable pour debug car pas d’augmentation aléatoire.

### `fast`

Même logique que `balanced`. Utilisé pour sampler vite beaucoup d’architectures.

### `no_bn_fast`

Désactive la calibration BN, mais garde le cache GPU pour l’évaluation.

Utile pour debug rapide, mais les labels d’accuracy des petits sous-réseaux peuvent être moins fiables.

---

## 9. Options d’évaluation une par une

### `cudnn_benchmark`

Active `torch.backends.cudnn.benchmark`.

Avantage : peut accélérer le GPU.

Inconvénient : moins déterministe ; peut ajouter un coût de benchmark si les shapes changent souvent.

### `cache_calib_on_gpu`

Précharge les batches de calibration BN en GPU.

Avantage : évite l’overhead DataLoader dans la boucle de sampling.

Inconvénient : consomme de la VRAM ; les mêmes batches sont réutilisés.

### `cache_eval_on_gpu`

Précharge les batches d’évaluation en GPU.

Avantage : accélère fortement `sample_acc_dataset`.

Inconvénient : mesure sur un subset fixe si `--val-batches` ou `--test-batches` est petit.

### `pre_resize_cache`

Précrée les versions 24×24, 28×28, 32×32 des images cachées.

Avantage : évite `F.interpolate` dans le forward.

Inconvénient : consomme plus de VRAM.

### `bn_calib_loader`

Choisit le loader utilisé pour la calibration BN.

Valeurs :

```text
train
val
```

`train` utilise les augmentations `RandomCrop` et `RandomHorizontalFlip`.

`val` utilise un split de train sans augmentation.

### `restore_bn_after_eval`

Sauvegarde/restaure les statistiques BN autour de l’évaluation.

Avantage : utile pour diagnostic.

Inconvénient : plus lent.

Dans `sample_acc_dataset`, ce n’est généralement pas nécessaire parce qu’on recalibre avant chaque subnet.

### `timing`

Affiche les temps par sample.

---

## 10. Surcharger les options d’évaluation

On peut passer un dictionnaire JSON en ligne de commande :

```bat
--eval-options "{\"cache_eval_on_gpu\": false, \"timing\": true}"
```

Ou donner un fichier JSON :

```bat
--eval-options ./my_eval_options.json
```

Exemple de fichier :

```json
{
  "cudnn_benchmark": true,
  "cache_calib_on_gpu": true,
  "cache_eval_on_gpu": true,
  "pre_resize_cache": true,
  "bn_calib_loader": "val",
  "restore_bn_after_eval": false,
  "timing": true
}
```

---

## 11. Référence des classes et fonctions

Cette section liste les fonctions importantes pour qu’un membre du groupe sache où modifier le code.

### 11.1 Helpers généraux

#### `set_seed(seed)`

Fixe les seeds Python et PyTorch.

Impact : les runs sont reproductibles si seed, dataset et hyperparamètres sont identiques.

#### `ensure_dir(path)`

Crée un dossier s’il n’existe pas.

#### `now_string()`

Retourne une date/heure sous forme string. Peu utilisé actuellement.

---

### 11.2 Système d’options d’évaluation

#### `EVAL_OPTION_PRESETS`

Dictionnaire central des presets d’évaluation.

C’est le bon endroit pour ajouter une nouvelle configuration expérimentale.

#### `parse_eval_options(preset, options_json)`

Charge un preset, applique éventuellement des overrides JSON, puis configure `torch.backends.cudnn.benchmark`.

#### `_select_images_for_spec(images, spec)`

Si les batches sont pré-resizés, `images` est un dictionnaire :

```python
{24: batch_24, 28: batch_28, 32: batch_32}
```

Cette fonction choisit le batch correspondant à `spec["r"]`.

#### `cache_batches_on_gpu(...)`

Précharge les batches sur GPU, éventuellement en plusieurs résolutions.

Utilisé pour accélérer la calibration BN et l’évaluation répétée de nombreux subnets.

#### `clone_bn_state(model)` et `restore_bn_state(model, state)`

Sauvegardent/restaurent seulement les statistiques BatchNorm.

Utiles si on veut vérifier que la calibration d’un subnet ne pollue pas l’état du modèle.

---

### 11.3 Search space

#### `SearchSpace`

Dataclass qui définit les choix possibles :

```python
resolutions = (24, 28, 32)
depth_choices = (2, 3, 4)
kernel_choices = (3, 5, 7)
expand_choices = (3, 4, 6)
n_stages = 5
blocks_per_stage = 4
```

#### `SearchSpace.largest_spec()`

Retourne le plus grand sous-réseau.

#### `SearchSpace.random_spec(...)`

Tire aléatoirement une architecture.

C’est une des fonctions principales à modifier pour tester d’autres stratégies de sampling.

#### `SearchSpace.sample_for_training_stage(stage)`

Définit les distributions de sampling pendant progressive shrinking.

Stages :

```text
largest
kernel
depth1
depth2
width1
width2/full
```

---

### 11.4 Couches dynamiques

#### `DynamicBatchNorm2d`

BatchNorm compatible avec un nombre dynamique de canaux.

Elle stocke une BN maximale, puis utilise seulement les `c` premiers canaux :

```python
running_mean[:c]
running_var[:c]
weight[:c]
bias[:c]
```

#### `ConvBNAct`

Bloc statique : convolution + BatchNorm + activation.

Utilisé pour le stem et la couche finale.

#### `DynamicPointConv2d`

Convolution 1×1 dynamique.

Slice les poids selon les canaux actifs.

#### `DynamicDepthwiseConv2d`

Depthwise convolution dynamique.

Gère :

```text
nombre de canaux actif
kernel size 3/5/7 par crop central
```

#### `DynamicMBConv`

Bloc MBConv dynamique :

```text
expand 1x1
BN + activation
depthwise dynamique
BN + activation
project 1x1
BN
skip connection éventuelle
```

#### `DynamicMBConv.sort_middle_channels_by_l1()`

Trie les canaux intermédiaires par importance L1.

Cela permet que les petits expansion ratios utilisent les canaux les plus importants en premier.

---

### 11.5 Réseau principal

#### `OFACifarNet`

Supernet complet.

Architecture :

```text
first_conv
first_block
5 stages × 4 DynamicMBConv
final_expand_layer
global average pooling
classifier
```

#### `set_active_subnet(spec)`

Active un sous-réseau précis.

#### `sample_active_subnet(stage)`

Tire un subnet selon la phase de training, puis l’active.

#### `set_largest_subnet()`

Active le plus grand subnet.

#### `forward(x, spec=None)`

Si `spec` est fourni, active ce subnet puis fait le forward.

#### `sort_all_middle_channels_by_l1()`

Applique le tri L1 à tous les blocs dynamiques.

---

### 11.6 Validation, encodage et MACs

#### `validate_spec(spec, ss)`

Vérifie qu’une spec est valide.

#### `spec_to_jsonable(spec)`

Convertit les valeurs en `int` purs pour pouvoir écrire en JSON.

#### `encode_spec(spec, ss)`

Transforme une architecture en vecteur one-hot pour le predictor.

Les blocs désactivés par la profondeur sont encodés avec des zéros.

#### `encoding_dim(ss)`

Retourne la dimension du vecteur d’entrée du MLP.

#### `conv2d_macs(...)`

Calcule les MACs d’une convolution.

#### `mbconv_macs(...)`

Calcule les MACs d’un bloc MBConv.

#### `estimate_macs_given_spec(spec, ss)`

Calcule les MACs analytiques d’un subnet.

C’est la fonction à remplacer si on veut passer à une lookup table de latence.

---

### 11.7 Données

#### `build_loaders(...)`

Construit trois loaders :

```text
train_loader : train avec augmentation
val_loader   : split train sans augmentation
 test_loader : test CIFAR-10 sans augmentation
```

Le split validation vient du train CIFAR-10, car CIFAR-10 n’a pas de validation officielle.

---

### 11.8 Entraînement et évaluation

#### `AverageMeter`

Accumule des moyennes pondérées.

#### `accuracy_top1(logits, target)`

Calcule l’accuracy top-1 en pourcentage.

#### `kd_loss(student_logits, teacher_logits, temperature)`

Loss de distillation KL.

#### `get_teacher_logits(model, images)`

Fait un forward du largest subnet du teacher figé.

#### `make_frozen_teacher(model, device)`

Crée une copie figée du modèle après l’entraînement largest.

Utilisée pour distiller les petits subnets pendant le shrinking.

#### `train_one_epoch(...)`

Une epoch d’entraînement du supernet.

Pour chaque batch :

```text
sample un ou plusieurs subnets
forward
CE loss + éventuellement KD loss
backward
optimizer.step
```

Le nombre de subnets par batch dépend du stage :

```text
largest/kernel : 1
depth1/depth2 : 2
width1/width2 : 4
```

#### `evaluate(...)`

Évalue un subnet sur un loader.

#### `reset_bn_stats(module)`

Reset toutes les running stats BatchNorm.

#### `calibrate_bn(...)`

Recalcule les stats BN pour un subnet donné avec momentum cumulatif.

---

### 11.9 Checkpoints

#### `save_checkpoint(...)`

Sauvegarde :

```text
model_state
optimizer_state
epoch
best_acc
search_space
active_spec
extra
```

#### `load_supernet_checkpoint(...)`

Recharge un supernet depuis un checkpoint.

---

### 11.10 Predictor

#### `AccuracyPredictor`

MLP architecture -> accuracy.

Par défaut :

```text
Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear(1)
```

#### `ArchAccDataset`

Dataset PyTorch qui lit le JSONL d’architectures.

#### `train_accuracy_predictor(...)`

Entraîne le MLP avec split 80/20 train/val.

Sauvegarde automatiquement le meilleur modèle selon `val_rmse`.

#### `load_accuracy_predictor(...)`

Recharge un predictor.

#### `predict_accuracy(...)`

Prédit l’accuracy d’une spec.

---

### 11.11 Sampling dataset

#### `sample_accuracy_dataset(...)`

Fonction centrale pour créer le dataset architecture/accuracy.

Pour chaque sample :

```text
1. tirer une spec
2. éviter les doublons avec la boucle de 100 tentatives
3. calculer les MACs
4. calibrer BN pour cette spec
5. évaluer cette spec
6. écrire une ligne JSONL
```

La boucle :

```python
for _ in range(100):
```

sert seulement à éviter de tirer deux fois exactement la même architecture.

---

### 11.12 Recherche évolutionnaire

#### `mutate_spec(...)`

Change aléatoirement certains gènes d’une spec.

#### `crossover_spec(...)`

Crée un enfant en mélangeant deux parents.

#### `evolutionary_search(...)`

Recherche génétique :

```text
1. créer une population initiale faisable
2. scorer avec predictor + MACs
3. garder les meilleurs parents
4. créer des enfants par mutation/crossover
5. répéter sur plusieurs générations
6. sauvegarder best_spec.json
```

---

### 11.13 Commandes CLI

#### `cmd_train_supernet`

Implémente la commande `train_supernet`.

#### `cmd_sample_acc_dataset`

Implémente la commande `sample_acc_dataset`.

#### `cmd_train_acc_predictor`

Implémente la commande `train_acc_predictor`.

#### `cmd_search`

Implémente la commande `search`.

#### `cmd_eval_subnet`

Implémente la commande `eval_subnet`.

#### `build_parser`

Définit tous les arguments CLI.

#### `main`

Parse les arguments et appelle la bonne commande.

---

## 12. Modifier les stratégies de sampling

### Sampling des subnets pour le predictor

Modifier :

```python
sample_accuracy_dataset(...)
```

et surtout la fonction interne :

```python
draw_spec()
```

Actuellement :

```text
uniform
small_biased
large_biased
```

Idées d’expériences :

```text
sampling stratifié par MACs
sampling équilibré par résolution
sampling actif selon incertitude du predictor
sur-échantillonnage près de la contrainte MACs
```

### Sampling pendant l’entraînement du supernet

Modifier :

```python
SearchSpace.sample_for_training_stage(stage)
```

C’est là que sont définies les distributions de sampling pendant progressive shrinking.

---

## 13. Points d’attention et pièges fréquents

### 13.1 `supernet_best.pt` vs `supernet_last.pt`

- `best` : bon pour le largest subnet.
- `last` : meilleur choix pour les subnets quelconques après progressive shrinking.

### 13.2 Calibration BN

Si l’accuracy tombe à 10%, tester :

```bat
--bn-calib-batches 0
```

pour voir si le checkpoint lui-même est OK.

Si largest marche sans calibration mais échoue avec calibration, problème dans BN calibration.

### 13.3 Windows et `num_workers`

Sous Windows, `--num-workers 0` peut être plus rapide que `4` pour CIFAR-10.

### 13.4 Predictor très rapide

C’est normal : le predictor est un petit MLP sur des vecteurs one-hot.

### 13.5 Search très rapide

C’est normal aussi : la recherche n’évalue pas les vrais CNN, seulement le MLP et les MACs analytiques.

---

## 14. Recommandations expérimentales

### Étude principale : nombre de samples predictor

Comparer :

```text
N = 100, 300, 500, 1000, 2000
```

Pour chaque N :

```text
1. créer acc_dataset_N.jsonl
2. entraîner predictor
3. lancer search
4. évaluer best_spec réel
5. comparer RMSE et accuracy finale
```

### Étude sampling strategy

Comparer :

```text
uniform
small_biased
large_biased
stratified à implémenter
```

### Étude budget MACs

Tester :

```text
40M, 60M, 80M, 100M, 120M
```

et construire une courbe Pareto.

---

## 15. Exemple de workflow complet conseillé

```bat
REM 1. Train supernet
python ofa_cifar10_options.py train_supernet ^
  --data-dir ./data ^
  --save-dir ./runs/ofa_cifar10 ^
  --epochs-largest 50 ^
  --epochs-kernel 25 ^
  --epochs-depth1 10 ^
  --epochs-depth2 25 ^
  --epochs-width1 10 ^
  --epochs-width2 25 ^
  --channel-sort ^
  --batch-size 128 ^
  --num-workers 0

REM 2. Sample 1000 architectures
python ofa_cifar10_options.py sample_acc_dataset ^
  --data-dir ./data ^
  --save-dir ./runs/ofa_cifar10 ^
  --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_last.pt ^
  --num-arch-samples 1000 ^
  --output ./runs/ofa_cifar10/acc_dataset_1000.jsonl ^
  --bn-calib-batches 5 ^
  --val-batches 10 ^
  --batch-size 512 ^
  --num-workers 0 ^
  --eval-preset accurate

REM 3. Train accuracy predictor
python ofa_cifar10_options.py train_acc_predictor ^
  --dataset ./runs/ofa_cifar10/acc_dataset_1000.jsonl ^
  --save-dir ./runs/ofa_cifar10 ^
  --predictor-epochs 200 ^
  --predictor-batch-size 64 ^
  --predictor-lr 0.001

REM 4. Search under 80M MACs
python ofa_cifar10_options.py search ^
  --predictor ./runs/ofa_cifar10/accuracy_predictor.pt ^
  --save-dir ./runs/ofa_cifar10 ^
  --macs-limit 80000000

REM 5. Evaluate selected subnet
python ofa_cifar10_options.py eval_subnet ^
  --data-dir ./data ^
  --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_last.pt ^
  --spec-json ./runs/ofa_cifar10/search/best_spec.json ^
  --bn-calib-batches 20 ^
  --batch-size 512 ^
  --num-workers 0 ^
  --eval-preset accurate
```

---

## 16. Notes de review rapide du code actuel

Le fichier compile correctement avec :

```bash
python -m py_compile ofa_cifar10_options.py
```

Quelques remarques :

1. Les imports `csv`, `math`, `os` semblent peu ou pas utilisés. Ce n’est pas bloquant.
2. `matplotlib.pyplot as plt` est importé, mais pas utilisé dans la version actuelle visible. Ce n’est pas bloquant.
3. Le docstring initial contient encore des exemples anciens (`--epochs-depth`, `--epochs-width`, nom `ofa_cifar10.py`). Le présent README donne les commandes à jour.
4. Le preset `accurate` contient quand même des optimisations par cache GPU. La précision réelle dépend surtout de `--bn-calib-batches`, `--val-batches` et `--test-batches`.
5. Pour une comparaison strictement “sans optimisation”, utiliser `--eval-preset no_opti`.

---

## 17. Résumé conceptuel en une phrase

Ce code entraîne un supernet OFA-like sur CIFAR-10, mesure un sous-ensemble de sous-réseaux, apprend un modèle rapide qui prédit leur accuracy, puis utilise ce predictor pour chercher efficacement une architecture sous contrainte de MACs.
