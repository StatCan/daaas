# Stockage

La plateforme propose différents types de stockage, conçus 
pour différents types de cas d’utilisation. Par conséquent, 
cette section vous concerne, que vous soyez en train d’expérimenter, 
de créer des pipelines, ou d’éditer.  

En surface, il existe deux types de stockage :

- des disques (aussi appelés volumes)

- des compartiments (stockage S3 ou "blob")


## Disques

![Volumes de données](images/kubeflow_existing_volume.png)

Les disques sont les systèmes de fichiers courants de type disque dur ou SSD. 
Vous pouvez monter les disques dans votre serveur Kubeflow, et même si vous 
supprimez votre serveur, vous pouvez remonter les disques, car ils ne sont jamais 
détruits par défaut. C’est un moyen très simple de stocker vos données, 
et si vous partagez un espace de travail avec une équipe, tous les membres peuvent 
utiliser le disque du même serveur (comme un lecteur partagé).


## Compartiments


![Compartiments/Stockage d’objets](images/minio_self_serve_bucket.png)

Les compartiments sont un peu plus compliqués, mais ils présentent trois avantages :

- Le stockage de grandes quantités de données
  - Les compartiments peuvent être énormes (bien plus grands que les disques durs), 
    et ils sont rapides.
  
- Le partage de données
  - Vous pouvez partager des fichiers à partir d’un compartiment en partageant 
    une URL que vouspouvez obtenir par l’intermédiaire d’une interface Web simple. 
    C’est une excellente façon de partager des données avec des personnes à 
    l'extérieur de votre espace de travail.
    
- L’accès à la programmation
  - Plus important encore, il est beaucoup plus facile pour les pipelines et 
    les navigateurs Web d’accéder aux données provenant de compartiments que 
    d’un disque dur. Donc, si vous voulez utiliser des pipelines, il faut d'abord 
    les configurer pour qu’ils fonctionnent avec un compartiment.
    

# Stockage en compartiment

Nous offrons quatre compartiments d’instances de stockage :

**Libre-service**

- [Minimal](https://minimal-tenant1-minio.example.ca)

- [Premium](https://premium-tenant1-minio.example.ca)

- [Pachyderm](https://pachyderm-tenant1-minio.example.ca)

**Accessible au grand public**

- [Public (en lecture seule)](https://datasets.example.ca)


## Libre-service

Dans chacune des trois options de libre-service, vous pouvez créer un compartiment personnel. 
Pour vous connecter, il vous suffit d’utiliser **OpenID** comme ci-dessous.


![Compartiments/Stockage d’objets](images/minio_self_serve_login.png)

Une fois que vous êtes connecté, vous pouvez créer un compartiment personnel 
selon le format `prenom.nom`. Voir la photo ci-dessous.


![Compartiments/Stockage d’objets](images/minio_self_serve_bucket.png)

## Partage

Vous pouvez facilement partager des fichiers individuels.

![Partage de fichiers Minio](images/minio_self_serve_share.png)


## Accès à la programmation

À FAIRE : Demander à Will et Zach
