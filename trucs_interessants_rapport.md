é# Notes d’architecture — Mini-OFA CIFAR-10

Ce document rassemble les points importants discutés pendant le développement du projet.  
Il est destiné à servir de base pour le rapport final, en particulier pour expliquer les choix d’architecture faits dans notre implémentation par rapport au papier **Once-for-All (OFA)** original.

---

## 1. Organisation générale de l’architecture

Le réseau implémenté est une version simplifiée d’OFA adaptée à CIFAR-10.

L’architecture globale est :

```text
Stem convolution
→ First MBConv fixe
→ 5 stages × 4 blocs MBConv élastiques
→ Convolution finale 1×1
→ Global Average Pooling
→ Classifier
```

Le papier OFA original s’appuie sur une architecture de type **MobileNetV3** entraînée sur ImageNet.  
Notre implémentation reprend les idées centrales du papier, mais avec plusieurs simplifications pour rendre le code plus lisible, plus stable et plus facile à modifier.

Les dimensions élastiques conservées sont :

- la résolution d’entrée ;
- la profondeur par stage ;
- la taille de kernel ;
- l’expansion ratio, qui joue ici le rôle principal de largeur interne.

---

## 2. Stages, units et blocs

Dans le code, on parle de **stages** :

```python
n_stages = 5
blocks_per_stage = 4
```

Chaque stage contient donc quatre blocs MBConv.  
Le réseau maximal contient :

```text
5 stages × 4 blocs = 20 blocs élastiques
```

Dans le papier OFA, le terme **unit** est souvent utilisé pour désigner ce que notre code appelle un **stage** : un groupe de blocs opérant à une même échelle de représentation, avec un nombre de channels de sortie fixé.

La correspondance est donc approximativement :

| Papier OFA | Code |
|---|---|
| unit | stage |
| block | bloc MBConv individuel |
| elastic depth | nombre de blocs actifs dans un stage |

Une spécification de sous-réseau contient toujours 20 valeurs pour `ks` et 20 valeurs pour `e`, car elle décrit le réseau maximal.  
Cependant, certains blocs peuvent être ignorés si la profondeur du stage est plus petite.

Exemple :

```python
spec = {
    "r": 32,
    "d": [2, 4, 3, 2, 4],
    "ks": [... 20 valeurs ...],
    "e":  [... 20 valeurs ...],
}
```

Ici :

```text
stage 0 : on garde les 2 premiers blocs sur 4
stage 1 : on garde les 4 blocs
stage 2 : on garde les 3 premiers blocs
stage 3 : on garde les 2 premiers blocs
stage 4 : on garde les 4 blocs
```

Les valeurs `ks` et `e` des blocs non actifs sont simplement ignorées.

---

## 3. Variation du nombre de channels

Le nombre de channels n’est pas constant dans tout le réseau.

Dans le code :

```python
self.stage_out_channels = [32, 48, 64, 96, 128]
```

Cela signifie que les channels augmentent au début de certains stages :

```text
24 → 32 → 48 → 64 → 96 → 128
```

Le changement de channels se fait principalement au **premier bloc de chaque stage**.  
Les blocs suivants du même stage gardent le même nombre de channels.

Exemple schématique :

```text
Stem : 24 channels

Stage 0:
24 → 32
32 → 32
32 → 32
32 → 32

Stage 1:
32 → 48
48 → 48
48 → 48
48 → 48
```

C’est une organisation classique dans les CNN modernes : on augmente progressivement la largeur du réseau à mesure que la résolution spatiale diminue.

---

## 4. MBConv : principe général

Les blocs utilisés sont des **MBConv**, pour *Mobile Inverted Bottleneck Convolution*.  
Ce type de bloc est utilisé dans MobileNetV2, MobileNetV3, EfficientNet et OFA.

Un MBConv suit la structure suivante :

```text
Input
→ Expansion 1×1
→ Depthwise convolution
→ Projection 1×1
→ Résidu éventuel
```

Dans le code, cela correspond à :

```python
self.expand_conv
self.depth_conv
self.project_conv
```

### 4.1 Expansion 1×1

L’expansion est une convolution 1×1 qui augmente le nombre de channels internes.

Si l’entrée a `C` channels et que l’expansion ratio vaut `e`, alors :

```text
mid_channels = C × e
```

Exemple :

```text
C = 32
e = 6
mid_channels = 192
```

Le bloc commence donc par projeter les activations dans un espace plus large.

### 4.2 Depthwise convolution

La depthwise convolution applique un filtre spatial indépendamment sur chaque channel.  
Contrairement à une convolution classique, elle ne mélange pas les channels entre eux.

C’est dans cette couche que la taille du kernel est élastique :

```text
3×3, 5×5 ou 7×7
```

### 4.3 Projection 1×1

La projection finale est aussi une convolution 1×1.  
Elle ramène le nombre de channels de l’espace intermédiaire vers le nombre de channels de sortie du bloc.

Exemple :

```text
192 channels → 32 channels
```

On parle de “projection” car on projette une représentation large vers une représentation plus compacte.

### 4.4 Résidu

Si les dimensions sont compatibles, le bloc ajoute l’entrée à la sortie :

```python
if self.stride == 1 and self.in_channels == self.out_channels:
    out = out + residual
```

Le bloc apprend donc une correction autour de l’identité :

```text
output = F(x) + x
```

Ce mécanisme stabilise l’optimisation, comme dans ResNet.

---

## 5. Expansion ratio et notion de width

Dans cette implémentation, la largeur élastique est représentée par l’**expansion ratio** `e`.

Le paramètre `e` contrôle le nombre de channels intermédiaires du MBConv :

```text
mid_channels = in_channels × e
```

Les valeurs possibles sont :

```python
expand_choices = (3, 4, 6)
```

Donc, pour un bloc avec 64 channels en entrée :

```text
e = 3 → mid_channels = 192
e = 4 → mid_channels = 256
e = 6 → mid_channels = 384
```

Plus `e` est grand, plus le bloc est large, expressif et coûteux en MACs.

Il faut insister sur le fait que cette largeur est **interne au MBConv**. Le bloc n’a pas pour sortie `e × in_channels`. La sortie du bloc est déterminée par `out_channels`, tandis que `e` contrôle seulement la phase d’expansion avant la depthwise convolution.

---

## 6. Clarification : ce que signifie “width” dans OFA

Un point important du papier OFA est que le mot **width** est un peu ambigu si on le lit avec l’intuition habituelle des CNN.

Le papier écrit d’abord que les unités du réseau ont progressivement :

```text
feature map size reduced
channel numbers increased
```

c’est-à-dire que, comme dans beaucoup de CNN, les stages plus profonds ont moins de résolution spatiale et plus de channels. C’est aussi ce que fait notre code avec :

```python
self.stage_out_channels = [32, 48, 64, 96, 128]
```

Mais quand le papier décrit l’espace de recherche élastique, il précise :

```text
the width expansion ratio in each layer is chosen from {3, 4, 6}
```

Dans le contexte d’un MBConv, ce **width expansion ratio** ne signifie pas que la sortie du bloc devient `3 × Cin`, `4 × Cin` ou `6 × Cin`. Il désigne la largeur interne du bloc.

Un MBConv a la forme :

```text
Cin
→ mid = e × Cin
→ Cout
```

Le facteur `e` agit donc sur le tenseur intermédiaire, pas directement sur la sortie du bloc. La projection finale 1×1 ramène ensuite l’activation vers `Cout`, qui est déterminé par l’architecture du backbone.

Ainsi, dans notre implémentation comme dans l’espace décrit dans OFA, l’élasticité de largeur est principalement représentée par :

```python
expand_choices = (3, 4, 6)
```

et donc par :

```text
mid_channels = in_channels × expansion_ratio
```

Il ne faut donc pas interpréter `e = 3` comme une multiplication cumulative des channels de sortie à chaque layer. Si c’était le cas, même le plus petit réseau exploserait très vite en nombre de channels :

```text
32 → 96 → 288 → 864 → ...
```

Ce n’est pas ce que font les MBConv. Le facteur d’expansion est temporaire : il élargit l’espace interne du bloc, puis la projection 1×1 revient à une largeur de sortie contrôlée par le backbone.

Conclusion importante pour le rapport : notre choix de rendre élastique l’expansion ratio est cohérent avec la façon dont OFA décrit son élasticité de largeur. Ce qui diffère surtout du papier n’est pas ce principe, mais les détails exacts du backbone MobileNetV3, les modules SE, les tailles de stages et les réglages d’entraînement.

---

## 7. h-swish

`h_swish` signifie **hard swish**.  
C’est une activation utilisée dans MobileNetV3.

Le swish original est :

```text
swish(x) = x · sigmoid(x)
```

mais il est relativement coûteux à calculer sur mobile.

Le hard-swish est une approximation plus efficace :

```text
h_swish(x) = x · ReLU6(x + 3) / 6
```

Dans le code, certaines couches utilisent ReLU, d’autres utilisent h-swish :

```python
self.stage_acts = ["relu", "relu", "relu", "h_swish", "h_swish"]
```

Ce choix est inspiré de MobileNetV3/OFA, où les activations h-swish sont utilisées dans les parties plus profondes du réseau.

---

## 8. Règles d’inclusion des sous-réseaux

OFA repose sur une contrainte forte : les petits sous-réseaux doivent être inclus dans les grands.

Cela se traduit par des règles déterministes :

### Profondeur

Dans chaque stage, on garde les premiers blocs.

```text
d = 2 → blocs 1 et 2 actifs
d = 3 → blocs 1, 2 et 3 actifs
d = 4 → blocs 1, 2, 3 et 4 actifs
```

On ne choisit jamais arbitrairement deux blocs parmi quatre.

### Kernel

Un kernel 3×3 est obtenu comme le centre d’un 5×5, lui-même centre d’un 7×7.

```text
3×3 ⊂ 5×5 ⊂ 7×7
```

### Expansion ratio

Un bloc avec un petit expansion ratio utilise les premiers channels de l’espace intermédiaire maximal.

Exemple :

```text
e = 6 → tous les channels internes
e = 3 → préfixe des channels internes
```

Cela suppose que les channels importants soient placés au début.

---

## 9. Channel sorting

Comme les sous-réseaux plus petits utilisent des préfixes de channels, il est important que les premiers channels soient les plus utiles.

Le code contient une opération de tri :

```python
sort_middle_channels_by_l1()
```

Elle calcule une importance approximative des channels à partir de la norme L1 des poids :

```text
importance(channel) = somme des valeurs absolues de ses poids
```

Puis elle réordonne les channels de manière décroissante d’importance.

Le tri doit être propagé à toutes les couches qui utilisent ces channels :

- convolution d’expansion ;
- BatchNorm d’expansion ;
- depthwise convolution ;
- BatchNorm depthwise ;
- projection finale.

Ce mécanisme est une version simplifiée du channel sorting d’OFA.

---

## 10. BatchNorm dynamique

Le code utilise une classe `DynamicBatchNorm2d`.

L’idée est de stocker une BatchNorm correspondant au nombre maximal de channels, puis de ne prendre qu’un préfixe lorsque le sous-réseau actif utilise moins de channels.

Exemple :

```python
self.bn.running_mean[:c]
self.bn.running_var[:c]
self.bn.weight[:c]
self.bn.bias[:c]
```

où `c` est le nombre de channels actifs.

Cela permet de partager une même couche BatchNorm entre plusieurs largeurs.

---

## 11. Problème conceptuel du BatchNorm partagé

Même si les poids peuvent être partagés par inclusion, les statistiques BatchNorm sont plus délicates.

Pour une couche BatchNorm, les statistiques idéales sont :

```text
running_mean ≈ moyenne des activations
running_var  ≈ variance des activations
```

Mais ces activations dépendent du sous-réseau actif.

Un petit sous-réseau et un grand sous-réseau n’ont pas exactement les mêmes distributions d’activations.  
Il n’y a donc aucune raison théorique pour que les bonnes statistiques du petit sous-réseau soient simplement les premières statistiques du grand sous-réseau.

C’est pour cela qu’il faut recalibrer les BatchNorm avant d’évaluer un sous-réseau.

---

## 12. Calibration BatchNorm

La calibration BatchNorm consiste à :

1. fixer un sous-réseau ;
2. réinitialiser les statistiques BatchNorm ;
3. passer quelques batches dans le modèle sans gradient ;
4. laisser les BatchNorm reconstruire leurs `running_mean` et `running_var` ;
5. évaluer ensuite le sous-réseau en mode `eval`.

Cette étape ne modifie pas les poids du réseau.  
Elle modifie seulement les statistiques BatchNorm.

Schéma :

```text
set_active_subnet(spec)
reset BN stats
model.train()
forward sur quelques batches sans backward
model.eval()
évaluation
```

---

## 13. Problème rencontré avec le momentum BatchNorm

Initialement, la calibration utilisait le momentum BatchNorm classique :

```text
momentum = 0.1
```

Après reset, les statistiques valent :

```text
running_mean = 0
running_var = 1
```

Avec un momentum fixe, ces valeurs initiales artificielles continuent d’influencer les statistiques recalculées, surtout si on utilise peu de batches.

Cela a provoqué des évaluations absurdes, par exemple une accuracy proche de 10% même pour le plus grand sous-réseau.

La correction consiste à utiliser une moyenne cumulative :

```text
momentum = 1 / (i + 1)
```

où `i` est l’indice du batch de calibration.

Ainsi :

```text
batch 1 → stats = stats du batch 1
batch 2 → stats = moyenne des batchs 1 et 2
batch 3 → stats = moyenne des batchs 1, 2 et 3
```

L’initialisation à 0/1 ne pollue plus l’estimation.

---

## 14. Sampling des architectures

Le dataset pour l’accuracy predictor est construit en tirant des sous-réseaux aléatoires.

Chaque sous-réseau est décrit par une `spec` :

```python
{
    "r": ...,
    "d": [...],
    "ks": [...],
    "e": [...]
}
```

Pour chaque `spec`, le code :

1. active le sous-réseau ;
2. calibre ses BatchNorm ;
3. évalue son accuracy sur quelques batches de validation ;
4. calcule ses MACs ;
5. écrit le résultat dans un fichier `.jsonl`.

Le nombre de sous-réseaux tirés est contrôlé par :

```bash
--num-arch-samples
```

Ce paramètre correspond à l’analogue du sampling des 16K architectures mentionné dans OFA.

---

## 15. Accuracy predictor

L’accuracy predictor est un petit MLP.

Son entrée est un encodage one-hot de l’architecture :

- résolution ;
- profondeur ;
- kernel size ;
- expansion ratio ;
- indicateur implicite des blocs actifs ou skipped.

Sa sortie est l’accuracy prédite en points de pourcentage.

Exemple :

```text
vraie accuracy = 73.2
accuracy prédite = 74.0
erreur = 0.8 point
```

Dans nos premiers tests, le predictor obtient une RMSE d’environ :

```text
0.77 point d’accuracy
```

Ce résultat est raisonnable pour un dataset de 1000 architectures.

---

## 16. Recherche évolutionnaire

La recherche finale est faite par un algorithme évolutionnaire.

Objectif :

```text
maximiser accuracy prédite
sous contrainte MACs ≤ budget
```

Chaque individu est une `spec`.

À chaque génération :

1. on trie la population selon l’accuracy prédite ;
2. on garde les meilleurs parents ;
3. on crée de nouveaux individus par mutation et crossover ;
4. on filtre selon la contrainte de MACs ;
5. on répète.

La recherche est très rapide, car elle n’évalue pas réellement les réseaux.  
Elle utilise seulement :

- le MLP predictor ;
- l’estimation analytique des MACs.

---

## 17. Différences majeures avec le papier OFA original

Cette implémentation est volontairement plus simple que le papier OFA.

| Aspect | OFA original | Notre implémentation |
|---|---|---|
| Dataset | ImageNet | CIFAR-10 |
| Backbone | MobileNetV3 complet | Mini MobileNet/OFA |
| Blocs | MBConv + SE + h-swish | MBConv simplifié |
| Width elasticity | Expansion ratio par layer `{3,4,6}` dans les MBConv | Expansion ratio par block `{3,4,6}` |
| Latence | Lookup table hardware | MACs analytiques |
| Sampling predictor | 16K architectures | Nombre configurable |
| Recherche | Spécialisation hardware | Search sous contrainte MACs |
| Objectif | Déploiement multi-hardware | Étude pédagogique et sampling |

---

## 18. Points importants à souligner dans le rapport

Plusieurs aspects méritent d’être discutés explicitement.

### 18.1 L’espace annoncé n’est pas un espace de graphes arbitraires

OFA ne choisit pas arbitrairement n’importe quels channels ou n’importe quels blocs.  
L’espace est structuré par des règles d’inclusion :

```text
petit sous-réseau ⊂ grand sous-réseau
```

Cela réduit la diversité réelle, mais rend le weight sharing possible.

### 18.2 Les BatchNorm sont une difficulté centrale

Les poids peuvent être partagés relativement naturellement, mais les statistiques BatchNorm dépendent fortement du sous-réseau actif.  
La calibration BN est donc indispensable pour obtenir des mesures fiables.

### 18.3 La “width” d’OFA est à interpréter via l’expansion ratio

Le papier parle d’**elastic width**, mais dans l’espace expérimental décrit, cette largeur est donnée par le **width expansion ratio** de chaque layer, choisi dans `{3,4,6}`.

Dans un MBConv, ce ratio contrôle la largeur intermédiaire :

```text
mid_channels = in_channels × expansion_ratio
```

Il ne faut donc pas comprendre que chaque layer multiplie sa sortie par 3, 4 ou 6. La sortie `Cout` est ramenée par la projection 1×1. Notre implémentation reprend cette idée : les sous-réseaux diffèrent par la largeur interne des MBConv.

### 18.4 Le predictor transforme un problème coûteux en problème rapide

Une fois le predictor entraîné, la recherche devient extrêmement rapide.  
Cela illustre l’intérêt central du pipeline OFA :

```text
payer un coût initial
→ obtenir une spécialisation quasi instantanée
```

### 18.5 Le nombre d’architectures samplées est un vrai paramètre expérimental

L’un des axes principaux du projet est de faire varier :

```text
100, 300, 1000, 2000, ...
```

architectures dans le dataset predictor, puis d’observer :

- RMSE du predictor ;
- qualité du subnet trouvé ;
- stabilité de la recherche.
