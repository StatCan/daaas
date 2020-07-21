# Vue d'ensemble

Kubeflow Pipelines est une plateforme de création de flux de production
d'apprentissage automatique pouvant être déployés dans un environnement
Kubernetes. Il permet de créer des pipelines qui encapsulent les flux de
production analytiques (transformation de données, modèles de formation,
construction d'éléments visuels, etc.). Ces pipelines peuvent être mis en
commun, réutilisés et programmés. Ils sont créés de façon à être exécutés avec
les calculs fournis par Kubernetes.

Dans le contexte de l'espace de travail en analytique avancée, vous pouvez
interagir avec les pipelines Kubeflow par l'entremise :

- de [l'interface utilisateur](../1-Experiments/Kubeflow.md) de Kubeflow, où, à
  partir du menu Pipelines, vous pouvez télécharger des pipelines, visualiser
  les pipelines que vous possédez et leurs résultats, etc.

- de la trousse [SDK](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/)
  en Python de Kubeflow Pipelines, accessible dans les serveurs de bloc-notes
  Jupyter, où vous pouvez définir vos composants et pipelines, les soumettre
  pour les exécuter immédiatement, ou même les enregistrer pour plus tard.

<!-- prettier-ignore -->
??? exemple "Des exemples supplémentaires dans les bloc-notes"
    Des exemples plus exhaustifs de pipelines produits expressément pour cette
    plateforme sont accessibles dans
    [GitHub](https://github.com/StatCan/jupyter-notebooks) (et dans chaque
    serveur de bloc-notes à /jupyter-notebooks). Vous pouvez également consulter
    des sources publiques.

Voyez
[le documentation officiel de Kubeflow](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)
pour obtenir une explication générale détaillée de Kubeflow.

![Un pipeline Kubeflow](../images/kf-pipeline_with_result.png)

# Qu'est-ce qu'un pipeline et comment fonctionne-t-il?

Dans Kubeflow Pipelines, un pipeline comprend un ou plusieurs composants de
pipeline enchaînés pour former un flux de production. Les composants sont comme
des fonctions que le pipeline connecte ensemble.

Le _pipeline_ décrit l'ensemble du flux de production de ce que vous souhaitez
accomplir, tandis que les _composants de pipeline_ décrivent chacune des étapes
distinctes de ce processus (comme le fait d'extraire des colonnes d'un stock de
données, de transformer des données ou d'entraîner un modèle). Chaque
**composant** doit être **modulaire** et idéalement **réutilisable**.

Essentiellement, chaque _composant_ a :

- une application autonome, présentée sous forme d'image de menu fixe
  (https://docs.docker.com/get-started/), pour effectuer le travail proprement
  dit .le code dans l'image de menu fixe peut être une séquence de commandes en
  langage naturel, un script Python ou tout autre code pouvant être exécuté à
  partir d'un terminal Linux
- une description de la manière dont Kubeflow Pipelines exécute le code
  (l'emplacement de l'image, les arguments de la ligne de commande qu'il
  accepte, les résultats qu'il produit), sous forme de fichier YAML.

Un _pipeline_ définit ensuite la logique de connexion des composants, par
exemple :

1. exécuter ComposantA
2. transmettre le résultat du ComposantA au ComposantB et au ComposantC
3. ...

<!-- prettier-ignore -->
??? example "Exemple d'un pipeline"
    Voici un exemple :

    ```python
    #!/bin/python3
    dsl.pipeline(name="Estimer Pi",
        description="Estimer Pi au moyen d'un modèle Map-Reduce")
    def compute_pi():
        # Créer un "exemple" d'opération pour chaque valeur de départ transmise au pipeline
        seeds = (1,2,3,4,5,6,7,8,9,10) sample_ops = [sample_op(seed) for seed in seeds]

        # Obtenir les résultats avant de les transmettre à deux pipelines distincts
        # Les résultats sont extraits des fichiers output_file.json et
        # sont disponibles à partir des instances sample_op par l'entremise de l'attribut .outputs
        outputs = [s.outputs['output'] for s in sample_ops]

        _generate_plot_op = generate_plot_op(outputs)
        _average_op = average_op(outputs)
    ```

    Vous pouvez trouver le pipeline complet dans
    [l'exemple `map-reduce-pipeline`](https://github.com/StatCan/jupyter-notebooks).

# Définir et exécuter votre premier pipeline

Bien que les _pipelines_ et les _composants_ soient définis par des fichiers
YAML, la trousse
[SDK](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/) en Python vous
permet de les définir à partir du code Python. Voici un exemple de définition
d'un
[pipeline simple](https://github.com/StatCan/jupyter-notebooks/blob/master/kfp-basics/average_with_docker_components.ipynb)
en utilisant la trousse SDK en Python.

L'objectif de notre pipeline est de calculer, au moyen de cinq nombres, les
valeurs suivantes :

1. la moyenne des trois premiers nombres;
2. la moyenne des deux derniers nombres;
3. la moyenne des résultats de (1) et de (2).

Pour ce faire, nous définissons un _pipeline_ qui utilise notre _composant_
moyen pour effectuer les calculs.

Le composant moyen est défini par une image de menu fixe au moyen d'un script
Python qui :

- accepte un ou plusieurs nombres comme arguments de ligne de commande
- renvoie la moyenne de ces nombres, enregistrée dans le fichier out.txt dans
  son conteneur.

Pour indiquer à Kubeflow Pipelines comment utiliser cette image, nous
définissons notre _composant_ moyen par l'entremise d'un ContainerOp, qui
indique à Kubeflow l'interface API de notre image. L'instance ContainerOp
définit l'emplacement de l'image du menu fixe, la façon de lui transmettre des
arguments et les résultats à extraire du conteneur. Pour utiliser réellement ces
ContainerOp dans notre pipeline, nous créons des fonctions de fabrique comme
average*op (car nous voudrons probablement plus d'un \_composant* moyen).

```python
from kfp import dsl

def average_op(\*numbers):
    """ Fabrique de ContainerOp moyen

    Accepte un nombre arbitraire de nombres d'entrée, en renvoyant un ContainerOp
    qui transmet ces nombres à l'image du menu fixe sous-jacent pour faire la
    moyenne

    Renvoie le résultat recueilli à partir du fichier ./out.txt à l'intérieur du
    conteneur

    """

    # Validation d'entrée

    if len(numbers) < 1:
        raise ValueError("Doit préciser au moins un nombre à partir duquel calculer la moyenne")

    return dsl.ContainerOp(
        name="averge", # Élément affiché dans la visionneuse de pipeline
        image="k8scc01covidacr.azurecr.io/kfp-components/average:v1", # L'image
        exécutée par Kuberflow Pipelines pour faire le travail arguments=numbers,
        #transmet chaque nombre comme un argument de ligne de commande (chaîne) distinct
        # Le script à l'intérieur du conteneur enregistre le résultat (sous forme de chaîne de caractères) dans le fichier out.txt, que
        # Kuberflow Pipelines lit pour nous et récupère sous forme de chaîne.
        file_outputs={'data': './out.txt'},
    )
```

Nous définissons notre pipeline comme une fonction Python qui utilise les
fabriques de ComponentOp ci-dessus, décorées par l'élément décoratif
@dsl.pipeline. Notre pipeline utilise notre composant _moyen_ en lui
transmettant des nombres. Puis, nous utilisons les résultats _moyens_ en les
transmettant à des fonctions plus tard par l'accès à `avg\_\*.output`.

```python
@dsl.pipeline(
    name="nom de mon pipeline"
)
def my_pipeline(a, b, c, d, e):
    """
    Calcul de moyenne de pipeline, qui accepte cinq nombres et effectue quelques calculs de moyenne sur ceux-ci
    """

    # Calculer les moyennes pour deux groupes

    avg_1 = average_op(a, b, c)
    avg_2 = average_op(d, e)

    # Utiliser les résultats de \_1 et de \_2 pour calculer une moyenne globale

    average_result_overall = average_op(avg_1.output, avg_2.output)
```

Enfin, nous enregistrons une définition YAML de notre pipeline pour la
transmettre plus tard à Kubeflow Pipelines. Ce fichier YAML décrit à Kubeflow
Pipelines exactement comment exécuter notre pipeline. Décompressez-le et voyez
par vous-même!

```python
from kfp import compiler
pipeline_yaml = 'pipeline.yaml.zip'
compiler.Compiler().compile(
    my_pipeline,
     pipeline_yaml
) print(f"Définition de pipeline exportée vers {pipeline_yaml}")
```

<!-- prettier-ignore -->
??? avertissement "Kubeflow Pipelines est une bête paresseuse".
    Il est utile de garder à l'esprit le calcul qui se produit lorsque vous
    exécutez ce code Python par rapport à ce qui se passe lorsque vous soumettez
    le pipeline à Kubeflow Pipelines. Bien que tout semble se produire
    instantanément, essayez d'ajouter `print(avg_1.output)` au pipeline
    ci-dessus et voyez ce qui se passe lorsque vous compilez votre pipeline. La
    trousse SDK en Python que nous utilisons sert à créer des pipelines, et non
    à les exécuter, de sorte que les résultats des composants ne seront jamais
    disponibles lorsque vous exécuterez ce code Python. Ce point est abordé plus
    en détail plus loin, dans la section _Comprendre l'ordre des calculs_.

Pour exécuter notre pipeline, nous définissons une expérience :

```python
experiment_name = "calcul de moyenne de pipeline"

import kfp
client = kfp.Client()
exp = client.create_experiment(name=experiment_name)

pl_params = { 'a': 5, 'b': 5, 'c': 8, 'd': 10, 'e': 18, }
```

Voici ce qui peut être observé dans
[l'interface utilisateur](../1-Experiments/Kubeflow.md) de Kubeflow Pipelines :

![Expérience Kubeflow Pipelines](../images/kfp_experiment.png)

Ensuite, nous exécutons une instance de notre pipeline en utilisant les
arguments souhaités :

```python
import time

run = client.run_pipeline(
    exp.id, # Exécuter dans l'expérience ci-dessus
    experiment_name + '-' + time.strftime("%Y%m%d-%H%M%S"), # Donner un nom et une
    heure système à notre tâche unique pipeline_yaml, # Transmettre le .yaml.zip que
    nous avons créé ci-dessus. Il définit le pipeline params=pl_params # Transmettre
    les paramètres en fonction desquels nous souhaitons exécuter le pipeline
)
```

Voici ce que l'on peut également voir dans l'interface utilisateur :

![Expérience Kubeflow Pipelines](../images/kfp_experiment.png)

Plus tard, lorsque nous souhaiterons réutiliser le pipeline, nous pourrons
transmettre différents arguments et tout recommencer (et même le réutiliser à
partir de l'interface utilisateur de Kubeflow). Pour mieux comprendre cet
exemple, ouvrez-le dans Kubeflow et essayez-le vous-même.

# Composants légers

En construction, malheureusement!

# Comprendre l'ordre des calculs

En construction, malheureusement!
