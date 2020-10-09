# Démarrer sur l'espace de travail d'analyse avancée

![Page d'accueil de l'espace de travail d'analyse avancée](images/readme/portal_ui.png)

Le
**[portail de l'espace de travail d'analyse avancée](https://portal.covid.cloud.statcan.ca)**
est un excellent endroit où explorer les ressources dont il sera question ici,
et y accéder.

Nous allons répartir les tâches standard en trois catégories :

1. **Expérimentation et analyse**
2. **Diffusion**
3. **Production à grande échelle**

Nous aborderons les trois, mais nous nous concentrerons sur les deux premières,
car elles sont les plus largement applicables.

# Pour l'expérimentation et l'analyse

<!--! [Kubeflow](images/logo-kubeflow.png){: style="max-height:200px"} -->

## Bloc-notes Jupyter

- `R`, `Python` et `Julia`
- Choisissez l'unité centrale ou la mémoire vive dont vous avez besoin, petite
  ou grande, pour votre analyse
- Partagez votre espace de travail avec votre équipe, y compris les données et
  les bloc-notes qui s'y trouvent

![Bloc-notes Jupyter](images/jupyter_in_action.png)

[**En savoir plus**](1-Experiences/Jupyter)

## Bureaux avec espace de travail ML

Il est plus facile de partager les bloc-notes que les bureaux, mais nous avons
aussi la possibilité d'exécuter un bureau complet, muni des applications
typiques, directement dans votre navigateur.

[**En savoir plus**](1-Experiences/ML-Workspaces)

# Pour la diffusion

## R-Shiny

<!-- prettier-ignore -->
![R-Shiny](images/logo-RStudio.png){:style="max-height:100px; display : block; margin-left : auto; margin-right : auto;"}

La plateforme est conçue pour accueillir le type d'application à source ouverte
de votre choix. Nous disposons d'un serveur R-Shiny pour l'hébergement des
applications R-Shiny

![Serveur R-Shiny](images/readme/shiny_ui.png)

Pour créer un tableau de bord R-Shiny, il suffit de soumettre une demande
d'extraction GitHub à notre
[répertoire R-Dashboards GitHub](https://github.com/StatCan/R-dashboards).

# Pour la production

Si une expérimentation se transforme en produit, l'un des éléments suivants peut
être nécessaire :

- des pipelines Kubeflow pour les travaux à haut volume ou haute intensité
- des pipelines d'automatisation

![Pipelines Kubeflow](images/readme/kubeflow_pipeline.png)

<!-- prettier-ignore -->
!!! info "Demander de l'aide pour la production"
    Le personnel de soutien de l'espace de travail d'analyse avancée est heureux
    de vous aider pour les cas d'utilisation orientés vers la production,
    et nous pouvons probablement vous faire gagner beaucoup de temps.
    N'hésitez pas à [nous demander pour de l'aide](Aide)!

# Comment obtenir des données? Comment envoyer des données?

![Parcourir les ensembles de données](images/readme/minio_ui.png)

- Chaque espace de travail peut être équipé de son propre stockage.

- Il existe également des compartiments de stockage pour la publication
  d'ensembles de données, pour usage interne ou diffusion plus large.

Nous donnerons un aperçu des technologies ici. Des renseignements plus précis
sur chacune d'entre elles seront fournis dans les sections suivantes.

<!-- prettier-ignore -->
!!! example "Parcourir quelques ensembles de données"
    Parcourez quelques [ensembles de données](https://datasets.covid.cloud.statcan.ca)
    ici. Ces ensembles de données ont été conçus pour stocker des données
    largement partagées. Il peut s'agir de données qui ont été introduites, ou
    de données qui seront diffusées sous forme de produit. **Comme toujours,
    veillez à ce qu'il ne s'agisse pas de données de nature délicate.**
