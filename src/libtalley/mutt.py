"""Email using mutt.

Mutt must be appropriately configured on the system and available on the PATH.
"""
import subprocess as sub


def send(to, subject, message, attach=None):
    """Send an email using mutt.
    
    Parameters
    ----------
    to : List[str]
        Email address(es) of the recipient(s).
    subject : str
        Subject line of the email.
    message : str
        The body of the message.
    attach : List[str], optional
        File(s) to attach. Can be a str if only one file.

    Returns
    -------
    process : subprocess.CompletedProcess
        Completed process information from running mutt.
    """
    if isinstance(to, str):
        to = [to]

    cmd = ['mutt', *to, '-s', subject]

    if attach is not None:
        if isinstance(attach, str):
            attach = [attach]
        cmd.extend(['-a', *attach, '--'])

    process = sub.run(cmd, input=message.encode())
    return process
