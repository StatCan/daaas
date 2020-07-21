# Collaboration on the Advanced Analytics Workspace

There are lots of ways to collaborate on the platform, and what's best for you
depends on _what you're sharing_ and _how many people you want to share with_.
We can roughly break the _sharable things_ into **Data** and **Code**, and we
can share the scope of who you're sharing with **No one** v.s. **My Team** v.s.
**Everyone**. This leads to the following table of options

|          |           **Private**            |           **Team**           |  **Statcan**  |
| :------: | :------------------------------: | :--------------------------: | :-----------: |
| **Code** | Gitlab/Github or personal folder | Gitlab/Github or team folder | Gitlab/Github |
| **Data** |    personal folder or bucket     |    team folder or bucket     | shared Bucket |

<!-- prettier-ignore -->
??? question "What is the difference between a bucket and a folder?"
    Buckets are like Network Storage. See the [Storage section](/Storage) section for more discussion of the differences between these two ideas.

The way that **Private** v.s. **Team** based access is configured is with
**namespaces**. So we start by talking about Kubeflow and Kubeflow namespaces.

Then, **Data** and **Code** are better handled with slightly different tools, so
we discuss the two seperately. With Data we discuss **buckets and Minio**, and
with Code we discuss **git**, **Gitlab**, and **Github**.

This motivates the structure of this page

- Team Based Collaboration (applicable to Code and Data)
- Sharing Code
- Sharing Data

# Team Based Collaboration

## What does Kubeflow do?

Kubeflow runs your **workspaces**. You can have notebook servers (called Jupyter
Servers), and in them you can create analyses in R and Python with interactive
visuals. You can save and upload data, download it, and your team can work along
side you.

## Requesting a namespace

By default, everyone gets their own _personal_ namespace, `firstname-lastname`.
But if you want to create a namespace for a team, then there is a button to
submit a request for a namespace on the portal. **Click the &#8942; menu on
[the kubeflow section of the portal](https://portal.covid.cloud.statcan.ca/#kubeflow)**.

<!-- prettier-ignore -->
!!! warning "The namespace cannot have special characters other than hypens"
    The namespace name must only be lowercase letters `a-z` with dashes. Otherwise,
    the namespace will not be created.

**You will receive an email notification when the namespace is created.** Once
the shared namespace is created, you can access it the same as any other
namespace you have through the Kubeflow UI, like shown below. You will then be
able to manage the collaborators list through the kubeflow **Manage
Contributors** tab, where you can add your colleagues to the shared namespace.

![Contributors Menu](/images/kubeflow_contributors.png)

To switch namespaces, take a look at the top of your window, just to the right
of the Kubeflow Logo.

![Select your Namespace](/images/kubeflow_manage_contributors.png)

# Shared Code

Teams have two options (but you can combine both!):

## Share a workspace in Kubeflow

The advantage of sharing inside Kubeflow is that it's more free-form and it
works better for `.ipynb` files (IPython Notebooks/Jupyter Notebooks). This
method also lets you share a compute environment, so you can share resources
very easily. When you share a workspace, you share

- A Private and Shared bucket (`/team-name` and `/shared/team-name`)
- All notebook servers in the Kubeflow Namespace

## Share with git, using Gitlab or Github

The advantage of sharing with git is that it works with users across namespaces,
and keeping code in git is a great way to manage large software projects.

<!-- prettier-ignore -->
!!! note "Don't forget to include a License!"
    If your code is public, do not forget to keep with the Innovation Team's
    guidelines and use a proper License if your work is done for Statistics
    Canada.

## Recommendation: Combine both

It's a great idea to always use git, and using git along with shared workspaces
is a great way to combine ad-hoc sharing (through files) while also keeping your
code organized and tracked.

# Shared Storage

## Sharing with your team

Once you have a shared namespace, you have two shared storage approaches

| Storage Option                           | Benefits                                                         |
| :--------------------------------------- | :--------------------------------------------------------------- |
| Shared Jupyter Servers/Workspaces        | More amenable to small files, notebooks, and little experiments. |
| Shared Buckets (see [Storage](/Storage)) | Better suited for use in pipelines, APIs, and for large files.   |

To learn more about the technology behind these, check out the
[Storage section](/Storage).

## Sharing with Statcan

In addition to private buckets, or team-shared private buckets, you can also
place your files in _shared storage_. Within all bucket storage options
(`minimal`, `premium`, `pachyderm`), you have a private bucket, **and** a folder
inside of the `shared` bucket. Take a look, for instance, at the link below:

- [`shared/blair-drummond/`](https://minimal-tenant1-minio.covid.cloud.statcan.ca/minio/shared/blair-drummond/)

Any **logged in** user can see these files and read them freely.

## Sharing with the world

Ask about that one in our [Slack channel](https://statcan-aaw.slaock.com). There
are many ways to do this from the IT side, but it's important for it to go
through proper processes, so this is not done in a "self-serve" way that the
others are. That said, it is totally possible.
