"""Generate the same startup check for untrusted agents and executable code."""
import re


def network_boundary_source(endpoints,*,subject='Executable',confirmations=1):
    pairs=[(str(host),int(port)) for host,port in endpoints]
    if not pairs or not all(re.fullmatch(r'[A-Za-z0-9_.:-]+',host) and 0<port<65536 for host,port in pairs):
        raise ValueError('Explicit network boundary endpoints are required')
    if type(confirmations) is not int or confirmations<1:
        raise ValueError('At least one boundary confirmation is required')
    return f'''
import socket,time
boundary_confirmations=0
while boundary_confirmations<{confirmations}:
    reachable=[]
    for host,port in {pairs!r}:
        try:
            with socket.create_connection((host,port),timeout=3):
                reachable.append(host)
        except OSError:
            pass
    if not reachable:
        boundary_confirmations+=1
    else:
        boundary_confirmations=0
        print({('Waiting for '+subject+' network boundary; untrusted code has not started: ')!r}+', '.join(reachable),flush=True)
    if boundary_confirmations<{confirmations}:
        time.sleep(5)
print({(subject+' network boundary verified before execution')!r},flush=True)
'''
