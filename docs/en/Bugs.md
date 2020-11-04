# Known Bugs

The Advanced Analytics Workspace combines many industry-best tools into one platform. The benefit of this is that we make use of some of the best technology and tools available; however one drawback is that sometimes we have to wait for fixes on the upstream software projects. The following lists a couple of known bugs, and their status:

- Unsupported Software
- In Progress Issues
- Upstream Issues


The **Unsupported Software** are mostly browser incompatibility issues. The solution for these is just to use another browser.


**In Progress Issues** are ones that we are actively working on, and hope to resolve on a certain timeline.


**Upstream Issues** are issues with the components such as MinIO or Kubeflow, where we have to wait for the projects themselves to get fixed.


# Unsupported Software

!!! failure "Internet Explorer (IE) is not supported"
    Internet Explorer simply does not support the web-technologies that many of these tools need,
    so it is impossible to make them work on IE. Firefox, Chromium, or (recent versions of) Microsoft Edge should all work.

!!! warning "Safari browser has limited support"
    Safari seems to support some functionality, but seems to have some incompatibilities with the platform. 
    We would recommend using Firefox, Chrome, or Microsoft Edge instead.

!!! note "Can I use SSH?"
    Unfortunately this is a security restriction; we cannot provide general access to SSH.

# In-Progress Issues

!!! warning "Incompatibility with GCCollaboration.ca"
    This unfortunate incompatibility is out of our control. The `@cloud.statcan.ca` domain has been around longer
    than `@GCCollaboration.ca`, and unfortunately that extra account is beyond our control. We are working on making the
    process to redirect to the correct `@cloud.statcan.ca` account simpler.

!!! failure "Cannot add users to Shared Namespace"
    We are working on this; it's just a configuration issue. We will try to get this working as soon as possible.

!!! note "Some issues with NetB"
    The platform was developed during a time where we were mostly off the VPN, so testing on NetB is now ongoing.
    Feel free to report any issues that you encounter on our Slack channel.

# Upstream issues

!!! warning "Cannot share files in MinIO when using OpenID Login"
    Unfortunately this is a [bug in MinIO](https://github.com/minio/minio/issues/8935). We might be able to work
    on this ourselves, but the issue is not very old, so hopefully it gets resolved before too long.
