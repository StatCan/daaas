# Collaboration en matière d'espace de travail en analytique avancée

Il existe de nombreuses façons de collaborer sur la plateforme, et ce qui vous
convient le mieux dépend des _éléments que vous partagez_ et du _nombre de
personnes avec qui vous souhaitez partager des éléments_. Nous pouvons répartir
les _éléments partageables_ en **données** et en **code**. Nous pouvons aussi
répartir l'ensemble des groupes avec lesquels vous partagez des éléments en
**privé**, **équipe** et **StatCan**. Cela conduit au tableau d'options suivant
:

|             |             **Privé**              |            **Équipe**             |     **StatCan**      |
| :---------: | :--------------------------------: | :-------------------------------: | :------------------: |
|  **Code**   | GitLab/GitHub ou dossier personnel | GitLab/GitHub ou dossier d'équipe |    GitLab/GitHub     |
| **Données** | Dossier ou compartiment personnel  | Dossier ou compartiment d'équipe  | Compartiment partagé |

<!-- prettier-ignore -->
??? question "Quelle est la différence entre un compartiment et un dossier?"
    Les compartiments s'apparentent au stockage réseau. Voir la
    [section sur le stockage](/Storage) pour obtenir des précisions
    supplémentaires sur les différences entre ces deux concepts.

Les accès **privé** et **équipe** sont configurés au moyen des **espaces de
nommage**. Nous commençons donc par parler de Kubeflow et des espaces de nommage
Kubeflow.

Ensuite, les **données** et le **code** sont mieux pris en charge au moyen
d'outils légèrement différents. C'est pourquoi nous allons les aborder en deux
temps. Pour les données, nous parlons de **compartiments et MinIO**. Pour le
code, nous parlons de **Git**, **GitLab** et **GitHub**.

C'est sur quoi repose la structure de cette page :

– Collaboration en équipe (applicable au code et aux données) – Partage du code
– Partage des données

# Collaboration en équipe

## Que fait Kubeflow?

Kubeflow exécute vos **espaces de travail**. Vous pouvez avoir des serveurs de
blocs-notes (appelés serveurs Jupyter) et vous pouvez y créer des analyses en R
et en Python en utilisant des éléments visuels interactifs. Vous pouvez
enregistrer, téléverser et télécharger des données. Votre équipe peut travailler
à vos côtés.

## Demande d'un espace de nommage

Par défaut, chaque personne obtient son _propre_ espace de nommage,
`prénom-nom`. Si vous souhaitez créer un espace de nommage pour une équipe,
présentez la demande dans le portail : **Cliquez sur le &#8942; de
[la section Kubeflow du portail](https://portal.covid.cloud.statcan.ca/#kubeflow)**.

<!-- prettier-ignore -->
!!! avertissement "L'espace de nommage ne doit comprendre aucun caractère spécial autre que le trait d'union."
    Le nom doit seulement comprendre des lettres minuscules (de « a » à « z »)
    et des traits d'union. Sinon, il sera impossible de créer l'espace de
    nommage.

**Vous recevrez un avis par courriel vous indiquant que l'espace de nommage est
créé.** Une fois que l'espace de nommage partagé est créé, vous pouvez y accéder
comme tous les autres espaces de nommage dont vous disposez par l'entremise de
l'interface utilisateur Kubeflow, comme il est indiqué ci-après. Vous pourrez
ensuite gérer la liste des collaborateurs sous l'onglet de **gestion des
contributeurs** de Kubeflow, où vous pourrez ajouter vos collègues à l'espace de
nommage partagé.

![Menu des contributeurs](/images/kubeflow_contributors.png)

Pour changer d'espace de nommage, allez en haut de votre fenêtre, juste à droite
du logo de Kubeflow.

![Sélectionnez votre espace de nommage](/images/kubeflow_manage_contributors.png)

# Code partagé

Les équipes ont deux options (mais vous pouvez combiner les deux) :

## Partager un espace de travail dans Kubeflow

Le partage dans Kubeflow présente l'avantage suivant : il est à structure libre
et il fonctionne mieux pour les fichiers `.ipynb` (blocs-notes Jupyter). Cette
méthode vous permet également de partager un environnement de calcul, de sorte
que vous pouvez partager des ressources très facilement. Lorsque vous partagez
un espace de travail, vous partagez :

– un compartiment privé et partagé (`/team-name` et `/shared/team-name`); – tous
les serveurs de blocs-notes dans l'espace de nommage Kubeflow.

## Partager avec Git, au moyen de GitLab ou de GitHub

Le partage au moyen de Git présente l'avantage suivant : il permet de travailler
avec des utilisateurs de tous les espaces de nommage. De plus, le fait de
conserver le code dans Git est un excellent moyen de gérer les grands projets de
logiciels.

<!-- prettier-ignore -->
!!! remarque "N'oubliez pas d'obtenir une licence d'utilisation!"
    Si votre code est public, suivez les directives de l'équipe de l'innovation
    et utilisez une licence appropriée si vous réalisez des tâches pour le
    compte de Statistique Canada.

## Recommandation : combinez les deux.

Il est sage de toujours utiliser Git. L'utilisation de Git de concert avec des
espaces de travail partagés est un bon moyen de combiner le partage ponctuel
(par l'entremise de fichiers) tout en assurant l'organisation et le suivi de
votre code.

# Stockage partagé

## Partage avec votre équipe

Une fois que vous avez un espace de nommage partagé, deux méthodes de stockage
partagé s'offrent à vous :

| Option de stockage                                 | Avantages                                                                                            |
| :------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| Serveurs/espaces de travail Jupyter partagés       | Mieux adaptés aux petits fichiers, aux blocs-notes et aux petites expériences.                       |
| Compartiments partagés (voir [Stockage](/Storage)) | Mieux adaptés à une utilisation dans les pipelines et les interfaces API et aux fichiers volumineux. |

Pour en savoir plus sur la technologie sous-jacente, consultez la
[section sur le stockage](/Storage).

## Partage avec StatCan

En plus des compartiments privés et des compartiments privés partagés en équipe,
vous pouvez également placer vos fichiers dans un espace de _stockage partagé_.
Toutes les options de stockage en compartiments (`minimal`, `supérieur`,
`éléphantesque`) offrent un compartiment privé **et** un dossier à l'intérieur
du compartiment `partagé`. Par exemple, consultez le lien ci-dessous :

- [`shared/blair-drummond/`](https://minimal-tenant1-minio.covid.cloud.statcan.ca/minio/shared/blair-drummond/)

Tous les utilisateurs **connectés** peuvent visualiser et consulter ces fichiers
librement.

## Partage avec le monde

Renseignez-vous à ce sujet sur notre
[canal Slack](https://statcan-aaw.slaock.com). Il existe de nombreuses méthodes
de partage avec le monde du point de vue informatique. Cependant, comme il est
important de respecter les processus appropriés, on n'utilise pas le
libre-service, comme dans les autres cas. Cela dit, c'est possible.
