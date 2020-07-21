# Pachyderm

<!-- prettier-ignore -->
!!! tip "Check for old volumes by looking at the Existing option"
    When you create your server you have the option of reusing an old volume
    or creating a new one. You probably want to reuse your old volume.

<!-- prettier-ignore -->
??? failure "Why am I getting "Missing url parameter: code"?"
    If you try to log into kubeflow and you get the message

    > Missing url parameter: code

    It is because you are signed in with the wrong Azure account. You must sign
    in with your cloud credentials.

    ![This means you're in the wrong account](../images/missing_parameter_code.png)
