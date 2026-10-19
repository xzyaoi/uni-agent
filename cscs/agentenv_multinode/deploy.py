#!/usr/bin/env python3
"""A configurable fleet of native AgentENV runtimes; mutations require --apply.

No launch is performed by default. State is private, atomic, and never contains
API keys. Keep the state directory: EC2 client tokens make interrupted launches
recoverable without allocating duplicate hosts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
import re
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
from datetime import datetime, timezone

DEFAULT_COUNT = 4
TYPE = 'm5d.metal'
AMI = 'ami-04643dd3d64c53242'
VPC = 'vpc-0bed9ca38b8cc7a62'
OLD_SG = 'sg-0ad4602cb5e252647'
PINNED_COMMIT = '6cccaa7842bd5be2051111d4f74d9e37aa721244'
STANDARD_FAMILY = re.compile(r'^(a|c|d|h|i|m|r|t|z)[0-9]')


def run(argv, *, data=None):
    result = subprocess.run(argv, input=data, text=True, capture_output=True)
    if result.returncode:
        # SSH scripts handle secrets internally: never include stdin in errors.
        raise RuntimeError(f'{argv[0]} failed ({result.returncode}): {result.stderr[-3000:]}')
    return result.stdout


def aws(args, service, command, *flags):
    return json.loads(run(['aws', '--region', args.region, '--output', 'json',
                           service, command, *flags]) or '{}')


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.chmod(0o600)
    temp.replace(path)


def inventory(args):
    # Stopping hosts retain their vCPU allocation until the stop completes.
    reservations = aws(args, 'ec2', 'describe-instances', '--filters',
                       'Name=instance-state-name,Values=pending,running,stopping')['Reservations']
    hosts = [host for reservation in reservations for host in reservation['Instances']]
    types = sorted({host['InstanceType'] for host in hosts} | {TYPE})
    shapes = aws(args, 'ec2', 'describe-instance-types', '--instance-types', *types)['InstanceTypes']
    cpus = {shape['InstanceType']: shape['VCpuInfo']['DefaultVCpus'] for shape in shapes}
    used = sum(cpus[host['InstanceType']] for host in hosts
               if STANDARD_FAMILY.match(host['InstanceType'].split('.')[0])
               and host.get('InstanceLifecycle') != 'spot')
    quota = aws(args, 'service-quotas', 'get-service-quota', '--service-code', 'ec2',
                '--quota-code', 'L-1216C47A')['Quota']['Value']
    ours = [host for host in hosts if any(tag['Key'] == 'AgentEnvExperiment'
            and tag['Value'] == args.name for tag in host.get('Tags', []))]
    required = max(0, args.count - len(ours)) * cpus[TYPE]
    old = aws(args, 'ec2', 'describe-security-groups', '--group-ids', OLD_SG)['SecurityGroups'][0]
    cidrs = set(args.allow_cidr)
    for permission in old['IpPermissions']:
        if permission.get('FromPort') in (22, 8000):
            cidrs.update(item['CidrIp'] for item in permission.get('IpRanges', [])
                         if item['CidrIp'] != '0.0.0.0/0')
    for cidr in cidrs:
        network = ipaddress.ip_network(cidr)
        if network.version != 4 or network.prefixlen == 0:
            raise ValueError('Only scoped IPv4 source CIDRs are allowed')
    return {'quota_vcpus': quota, 'used_standard_vcpus': used,
            'additional_vcpus': required, 'quota_ok': used + required <= quota,
            'source_cidrs': sorted(cidrs), 'instances': ours}


def ssh(args, host, script, payload=None):
    command = 'sudo python3 -c ' + shlex.quote(script)
    return run(['ssh', '-i', str(args.identity.expanduser()), '-o', 'IdentitiesOnly=yes',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                '-o', 'StrictHostKeyChecking=accept-new',
                'ubuntu@' + host['PublicDnsName'], command],
               data=json.dumps(payload) if payload is not None else None)


def require_apply(args):
    if not args.apply:
        raise ValueError('This command mutates infrastructure; add --apply after reviewing plan.json')


def provision(args, state, info):
    require_apply(args)
    if len(info['instances']) not in (0, args.count):
        raise RuntimeError('Partial tagged fleet exists: inspect/reconcile explicitly; no further launches')
    if len(info['instances']) == args.count:
        state['instance_ids'] = sorted(host['InstanceId'] for host in info['instances'])
        return
    if not info['quota_ok']:
        raise RuntimeError('Insufficient standard On-Demand vCPU quota; no resources were created')
    if not info['source_cidrs']:
        raise ValueError('No allowed client/SSH CIDRs found; supply --allow-cidr')
    aws(args, 'ec2', 'describe-images', '--image-ids', args.ami)
    aws(args, 'iam', 'get-instance-profile', '--instance-profile-name', 'agentenv-host')
    aws(args, 'ec2', 'describe-key-pairs', '--key-names', 'agentenv-xiayao')
    groups = aws(args, 'ec2', 'describe-security-groups', '--filters',
                 'Name=vpc-id,Values=' + VPC,
                 'Name=group-name,Values=' + args.name)['SecurityGroups']
    if groups:
        sg = groups[0]['GroupId']
    else:
        sg = aws(args, 'ec2', 'create-security-group', '--vpc-id', VPC,
                 '--group-name', args.name, '--description', 'AgentENV multinode runtime experiment')['GroupId']
    state['security_group_id'] = sg
    save(args.state / 'state.json', state)
    permissions = [{'IpProtocol': 'tcp', 'FromPort': port, 'ToPort': port,
                    'UserIdGroupPairs': [{'GroupId': sg}]} for port in (8000, 9090)]
    permissions += [{'IpProtocol': 'tcp', 'FromPort': port, 'ToPort': port,
                     'IpRanges': [{'CidrIp': cidr} for cidr in info['source_cidrs']]}
                    for port in (22, 8080)]
    for permission in permissions:
        # AWS detects duplicate rules; only the explicit duplicate error is benign.
        try:
            aws(args, 'ec2', 'authorize-security-group-ingress', '--group-id', sg,
                '--ip-permissions', json.dumps([permission]))
        except RuntimeError as error:
            if 'InvalidPermission.Duplicate' not in str(error):
                raise
    subnets = aws(args, 'ec2', 'describe-subnets', '--filters', 'Name=vpc-id,Values=' + VPC,
                  'Name=default-for-az,Values=true')['Subnets']
    subnet = args.subnet or sorted(subnets, key=lambda item: item['AvailabilityZone'])[0]['SubnetId']
    state.setdefault('client_token', secrets.token_hex(24))
    state['subnet_id'] = subnet
    save(args.state / 'state.json', state)
    launched = aws(args, 'ec2', 'run-instances', '--image-id', args.ami,
                   '--instance-type', TYPE, '--count', str(args.count),
                   '--client-token', state['client_token'], '--key-name', 'agentenv-xiayao',
                   '--subnet-id', subnet, '--security-group-ids', sg,
                   '--iam-instance-profile', 'Name=agentenv-host',
                   '--block-device-mappings', json.dumps([{'DeviceName': '/dev/sda1', 'Ebs': {
                       'VolumeSize': 350, 'VolumeType': 'gp3', 'DeleteOnTermination': True}}]),
                   '--metadata-options', 'HttpTokens=required,HttpPutResponseHopLimit=1',
                   '--user-data', 'AENV_S3_BUCKET=swe-bench-pro-templates\nAENV_S3_PREFIX=templates/\nAENV_S3_REGION=' + args.region + '\n',
                   '--tag-specifications', json.dumps([{'ResourceType': 'instance', 'Tags': [
                       {'Key': 'Name', 'Value': args.name},
                       {'Key': 'AgentEnvExperiment', 'Value': args.name}]}]))
    state['instance_ids'] = sorted(host['InstanceId'] for host in launched['Instances'])
    state['phase'] = 'booting_needs_runtime_inspection'
    save(args.state / 'state.json', state)


INSPECT = r'''
import hashlib,json,os,subprocess,shlex
p=json.load(__import__('sys').stdin)
def out(cmd):
 r=subprocess.run(cmd,text=True,capture_output=True)
 return {'returncode':r.returncode,'stdout':r.stdout[-6000:]}
binary='/usr/local/bin/server'
api_addr=None
if os.path.isfile('/etc/default/aenv'):
 for line in open('/etc/default/aenv'):
  if line.startswith('API_ADDR='):
   tokens=shlex.split(line.split('=',1)[1].strip())
   api_addr=tokens[0] if tokens else None
print(json.dumps({'runtime_sha256':hashlib.sha256(open(binary,'rb').read()).hexdigest() if os.path.isfile(binary) else None,
 'source_commit':out(['git','-C',p['source'],'rev-parse','HEAD']),
 'kernel':out(['uname','-r']),'kvm':os.path.exists('/dev/kvm'),
 'service':out(['systemctl','show','aenv','-p','ExecStart','-p','FragmentPath']),
 'health':out(['curl','-fsS','http://127.0.0.1:8000/health']),
 'listeners':out(['ss','-ltnp']),
 'runtime_api_addr':api_addr,
 'recorder_active':out(['systemctl','is-active','aenv-recorder']),
 'storage_config_present':os.path.isfile('/etc/default/aenv')}))
'''

CONFIGURE = r'''
import hashlib,json,os,pathlib,subprocess,sys
p=json.load(sys.stdin)
def call(a): subprocess.run(a,check=True,stdout=subprocess.DEVNULL)
def text(a): return subprocess.check_output(a,text=True).strip()
assert hashlib.sha256(pathlib.Path('/usr/local/bin/server').read_bytes()).hexdigest()==p['runtime_sha256'], 'runtime binary changed'
assert text(['git','-C',p['source'],'rev-parse','HEAD'])==p['commit'], 'source commit mismatch'
assert not text(['git','-C',p['source'],'status','--porcelain']), 'source checkout is dirty'
env=pathlib.Path('/etc/default/aenv')
assert env.exists(), 'AMI runtime environment absent'
root=pathlib.Path('/etc/agentenv');root.mkdir(mode=0o750,exist_ok=True)
backup=root/'aenv.pre-multinode'
if not backup.exists(): backup.write_bytes(env.read_bytes());backup.chmod(0o600)
values={'API_ADDR':p['runtime_api_addr'],'AENV_NODE_ID':p['node_id'],
'AENV_OBSERVABILITY_SCHEDULER_REPORT_ENABLED':'true',
'AENV_OBSERVABILITY_SCHEDULER_ENDPOINT':'http://'+p['control']+':9090',
'AENV_API_KEY':p['api_key'],'AENV_SANDBOX_ACCESS_TOKEN_HASH_SEED':p['seed']}
lines=[line for line in env.read_text().splitlines() if line.split('=',1)[0] not in values]
lines.extend(k+'="'+v+'"' for k,v in values.items())
env.write_text('\n'.join(lines)+'\n');env.chmod(0o600)
if p['is_control']:
 call(['make','-C',p['source']+'/services','build'])
 for name in ('scheduler','gateway'):
  call(['install','-m','0755',p['source']+'/services/bin/'+name,'/usr/local/bin/agentenv-'+name])
 if subprocess.run(['id','agentenv-control'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
  call(['useradd','--system','--no-create-home','--shell','/usr/sbin/nologin','agentenv-control'])
 auth=root/'auth.env';auth.write_text('AENV_API_KEY='+p['api_key']+'\n');auth.chmod(0o640)
 config=root/'control-plane.json';config.write_text(json.dumps(p['config'],indent=2));config.chmod(0o640)
 call(['chown','root:agentenv-control',str(root),str(auth),str(config)])
 for name in ('scheduler','gateway'):
  body='[Unit]\nDescription=AgentENV '+name+'\nAfter=network-online.target\nWants=network-online.target\n[Service]\nUser=agentenv-control\nGroup=agentenv-control\n'
  if name=='gateway': body+='EnvironmentFile=/etc/agentenv/auth.env\n'
  body+='ExecStart=/usr/local/bin/agentenv-'+name+' -config /etc/agentenv/control-plane.json\nRestart=on-failure\nRestartSec=5\nNoNewPrivileges=true\n[Install]\nWantedBy=multi-user.target\n'
  pathlib.Path('/etc/systemd/system/agentenv-'+name+'.service').write_text(body)
 call(['systemctl','daemon-reload'])
 call(['systemctl','enable','--now','agentenv-scheduler','agentenv-gateway'])
 call(['systemctl','restart','agentenv-scheduler','agentenv-gateway'])
call(['systemctl','restart','aenv'])
print(json.dumps({'configured':p['node_id']}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'provision', 'inspect', 'configure', 'status'])
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--name', default='agentenv-pro-4x')
    parser.add_argument('--count', type=int, default=DEFAULT_COUNT,
                        help='Number of runtime hosts (default: 4); part of the persisted fleet identity')
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--ami', default=AMI)
    parser.add_argument('--subnet')
    parser.add_argument('--allow-cidr', action='append', default=[])
    parser.add_argument('--identity', type=Path, default=Path('~/.ssh/agentenv-xiayao.pem'))
    parser.add_argument('--source', default='/sandbox/AgentENV')
    parser.add_argument('--compatibility', type=Path,
                        help='Reviewed provenance JSON: commit, runtime_sha256, heartbeat_supported=true, shared_storage_verified=true')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be a positive integer')
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.state.chmod(0o700)
    with (args.state / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.state / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'name': args.name, 'region': args.region, 'ami': args.ami, 'count': args.count,
            'created_at': datetime.now(timezone.utc).isoformat(), 'phase': 'planned'}
        # State written before --count always described the original four-host fleet.
        state.setdefault('count', DEFAULT_COUNT)
        if any(state[key] != value for key, value in [('name',args.name),('region',args.region),
                                                     ('ami',args.ami),('count',args.count)]):
            raise ValueError('State identity mismatch; use the original arguments')
        info = inventory(args)
        plan = {key: value for key, value in info.items() if key != 'instances'}
        plan.update({'runtime_count': args.count, 'instance_type': TYPE, 'ami': args.ami,
                     'gateway_cohost': True, 'public_port': 8080,
                     'private_ports': [8000,9090], 'gateway_request_timeout': '3600s',
                     'shared_storage': 's3://swe-bench-pro-templates/templates/',
                     'instance_ids': [host['InstanceId'] for host in info['instances']]})
        save(args.state / 'plan.json', plan)
        if args.command == 'provision':
            provision(args, state, info)
        elif args.command in ('inspect', 'configure'):
            ids = state.get('instance_ids', [])
            if len(ids) != args.count:
                raise ValueError(f'Provision all {args.count} hosts first')
            response = aws(args, 'ec2', 'describe-instances', '--instance-ids', *ids)
            hosts = sorted([host for reservation in response['Reservations'] for host in reservation['Instances']],
                           key=lambda host: host['InstanceId'])
            if sorted(host['InstanceId'] for host in hosts) != sorted(ids):
                raise ValueError('AWS instance response differs from the persisted fleet')
            state['hosts'] = [{'instance_id': h['InstanceId'], 'node_id': 'runtime-'+str(i+1),
                              'private_ip': h['PrivateIpAddress'], 'public_ip': h.get('PublicIpAddress'),
                              'public_dns': h['PublicDnsName']} for i,h in enumerate(hosts)]
            if args.command == 'inspect':
                reports = {h['InstanceId']: json.loads(ssh(args,h,INSPECT,{'source':args.source})) for h in hosts}
                save(args.state / 'runtime-inspection.json', reports)
                state['phase'] = 'needs_reviewed_runtime_compatibility'
            else:
                require_apply(args)
                if not args.compatibility:
                    raise ValueError('Reviewed --compatibility provenance is required; never upgrade AMI binaries blindly')
                provenance = json.loads(args.compatibility.read_text())
                if not (provenance.get('heartbeat_supported') is True and provenance.get('shared_storage_verified') is True
                        and provenance.get('runtime_api_addr') and isinstance(provenance.get('recorder_active'), bool)):
                    raise ValueError('Establish heartbeat/shared-storage compatibility and reviewed runtime/recorder topology first')
                runtime_api_addr = provenance['runtime_api_addr']
                recorder_active = provenance['recorder_active']
                expected_port = '8001' if recorder_active else '8000'
                if runtime_api_addr not in ('0.0.0.0:'+expected_port, '127.0.0.1:'+expected_port):
                    raise ValueError('Reviewed API address must match the recorder topology')
                if not recorder_active and runtime_api_addr != '0.0.0.0:8000':
                    raise ValueError('Direct runtime must accept private gateway traffic on 8000')
                commit, digest = provenance['commit'], provenance['runtime_sha256']
                if commit != PINNED_COMMIT or len(digest) != 64:
                    raise ValueError('Require stable v0.2.3 pinned commit and runtime SHA256')
                reports = {h['InstanceId']: json.loads(ssh(args,h,INSPECT,{'source':args.source})) for h in hosts}
                save(args.state / 'runtime-inspection.json', reports)
                for report in reports.values():
                    if report['runtime_sha256'] != digest or report['source_commit']['stdout'].strip() != commit or not report['kvm'] or report['health']['returncode'] != 0:
                        raise ValueError('All runtimes must pass hash, source commit, KVM and health checks before configuration')
                    if report['runtime_api_addr'] != runtime_api_addr or (report['recorder_active']['returncode'] == 0) != recorder_active:
                        raise ValueError('Runtime API address/recorder state differs from reviewed topology')
                secret_path = args.state / 'secrets.json'
                if not secret_path.exists():
                    save(secret_path, {'api_key':'e2b_'+secrets.token_hex(32),'seed':secrets.token_hex(32)})
                secret = json.loads(secret_path.read_text())
                control = hosts[0]['PrivateIpAddress']
                config = {'log_level':'info','log_format':'json', 'scheduler': {
                    'grpc_listen_addr':control+':9090','metrics_listen_addr':'127.0.0.1:9101',
                    'strategy':'round_robin','report_ttl':'30s','binding_ttl':'30s',
                    'discovery':{'mode':'static'},'nodes':[{'id':'runtime-'+str(i+1),
                        'endpoint':'http://'+h['PrivateIpAddress']+':8000'} for i,h in enumerate(hosts)]},
                    'gateway': {'http_listen_addr':'0.0.0.0:8080','metrics_listen_addr':'127.0.0.1:9102',
                        'scheduler_addr':control+':9090','request_timeout':'3600s',
                        'forward_response_size':4194304,'sandbox_proxy_domains':[]}}
                save(args.state / 'control-plane.json', config)
                for i, host in enumerate(hosts):
                    ssh(args, host, CONFIGURE, dict(secret, source=args.source, commit=commit,
                        runtime_sha256=digest, runtime_api_addr=runtime_api_addr, control=control,node_id='runtime-'+str(i+1),
                        is_control=i==0,config=config))
                    state.setdefault('configured_ids', [])
                    if host['InstanceId'] not in state['configured_ids']:
                        state['configured_ids'].append(host['InstanceId'])
                    save(path,state)
                state['gateway_url'] = 'http://'+hosts[0]['PublicDnsName']+':8080'
                state['phase'] = 'configured_needs_heartbeat_auth_and_sdk_readiness'
                state['gpu_launch_allowed'] = False
        save(path,state)
        manifest = {'schema_version': 1, 'region': args.region, 'version': 'v0.2.3',
                    'source_commit': PINNED_COMMIT, 'instances': state.get('hosts', []),
                    'gateway_endpoint': state.get('gateway_url'),
                    'secret_file': str((args.state / 'secrets.json').resolve()),
                    'readiness_state': state['phase'],
                    'gpu_launch_allowed': state.get('gpu_launch_allowed', False),
                    'shared_storage': 's3://swe-bench-pro-templates/templates/'}
        save(args.state / 'cluster.json', manifest)
        print(json.dumps({'state_file':str(path),'phase':state['phase'],'plan':plan}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
