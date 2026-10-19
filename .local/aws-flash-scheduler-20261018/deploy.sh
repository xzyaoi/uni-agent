#!/usr/bin/env bash
set -euo pipefail
commit=7d9cba5d328a12632e0a0e29297f4a4891a02e37
release=/sandbox/flash-sandbox/releases/uni-agent-7d9cba5d328a-20261018
receipt=/sandbox/uni-agent-deploy-20261018
old_release=$(readlink -f /sandbox/flash-sandbox/current)
[[ ! -e "$release" ]] || { echo 'Release already exists; inspect before retrying'; exit 1; }
mkdir -p "$receipt" "$release/bin" "$release/guest-assets"
printf '%s\n' "$old_release" > "$receipt/previous-release"
cp /usr/local/bin/flash-sandbox-scheduler "$receipt/previous-scheduler-wrapper"
systemctl stop flash-sandbox-scheduler.service flash-sandbox.service
python3 - <<'PY'
import socket
for port in (8080,8081,8090,8091):
    with socket.socket() as sock:
        sock.settimeout(1)
        assert sock.connect_ex(('127.0.0.1',port)) != 0, f'Old listener remains on {port}'
PY
tar -xzf /tmp/binaries.tar.gz -C "$release"
python3 - "$release" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for name,expected in json.load(open('/tmp/binary-sha256.json')).items():
    assert hashlib.sha256((root/'bin'/name).read_bytes()).hexdigest()==expected,name
print('Latest host binaries verified',flush=True)
PY
cp "$old_release/config.yaml" "$release/config.yaml"
python3 - "$release" <<'PY'
import pathlib,re,sys
root=pathlib.Path(sys.argv[1]);config=root/'config.yaml'
text=config.read_text()
for key in ('base_rootfs_path','python_base_path'):
    match=re.search(r'^  '+key+r': (.+)$',text,re.M)
    assert match,key
    old=pathlib.Path(match.group(1).strip())
    assert old.is_file(),old
    new=root/'guest-assets'/old.name
    import subprocess
    subprocess.run(['cp','--reflink=auto',str(old),str(new)],check=True)
    text=text.replace(match.group(0),'  '+key+': '+str(new))
config.write_text(text)
PY
mount_dir=$(mktemp -d /mnt/flash-envd-update.XXXXXX)
trap 'mountpoint -q "$mount_dir" && umount "$mount_dir"; rmdir "$mount_dir"' EXIT
for filesystem in "$release/guest-assets/base-rootfs.ext4" "$release/guest-assets/rootfs-alpine.ext4"; do
    mount -o loop "$filesystem" "$mount_dir"
    install -D -m 755 "$release/bin/flash-envd-linux-amd64" "$mount_dir/usr/local/bin/flash-envd"
    if [[ "$filesystem" == */base-rootfs.ext4 ]]; then
        install -D -m 755 /tmp/overlay-init "$mount_dir/sbin/overlay-init"
    fi
    cmp "$release/bin/flash-envd-linux-amd64" "$mount_dir/usr/local/bin/flash-envd"
    sync -f "$mount_dir/usr/local/bin/flash-envd"
    umount "$mount_dir"
    e2fsck -fn "$filesystem" > "$receipt/$(basename "$filesystem").fsck.txt" 2>&1
done
rmdir "$mount_dir"
trap - EXIT
echo 'Latest guest agents installed; cached container images preserved'
cp -a /opt/flash-sandbox/scheduler "$receipt/previous-scheduler"
cp -a /opt/flash-sandbox/sdk/python "$receipt/previous-sdk"
mv /opt/flash-sandbox/scheduler /opt/flash-sandbox/scheduler.before-20261018
tar -xzf /tmp/python-source.tar.gz -C /opt/flash-sandbox
/opt/flash-sandbox/.venv/bin/python -m pip install --disable-pip-version-check --no-deps /opt/flash-sandbox/sdk/python > "$receipt/sdk-install.log" 2>&1
cd /opt/flash-sandbox
/opt/flash-sandbox/.venv/bin/python -c 'import flash_sandbox; from scheduler.cli import parse_args; parse_args(["--cluster-url","http://127.0.0.1:8080","--resident","--ui","--no-cpu-scaling"]); print("Latest SDK and scheduler import successfully")'
cat > "$release/run-cluster.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$release'
exec ./bin/cluster --http-addr 127.0.0.1:8080 --scheduler random-image-affinity >> /sandbox/flash-sandbox/cluster.log 2>&1
EOF
cat > "$release/run-node.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$release'
exec sudo -n ./bin/node --http-port 8081 --coordinator-addr 127.0.0.1:8080 --advertise-addr 127.0.0.1:8081 --containerd-address /run/containerd/containerd.sock --config ./config.yaml --template-dir '$receipt/templates' --capacity-disk-path /sandbox --default-cpu-cores 4 --image-gc-interval 0s --reap-orphans-on-start=false --cleanup >> /sandbox/flash-sandbox/node.log 2>&1
EOF
cp "$old_release/run-stop.sh" "$release/run-stop.sh"
chmod 755 "$release"/run-*.sh
printf 'git_commit=%s\nbuilt_at=%s\nsource_ami=ami-0d04783cd08f6525a\n' "$commit" "$(date -u +%FT%TZ)" > "$release/VERSION"
sha256sum "$release"/bin/* "$release"/guest-assets/*.ext4 > "$receipt/deployed-sha256.txt"
chown -R ubuntu:ubuntu "$release" "$receipt" /opt/flash-sandbox/scheduler /opt/flash-sandbox/sdk
ln -s "$release" /sandbox/flash-sandbox/current.next
mv -Tf /sandbox/flash-sandbox/current.next /sandbox/flash-sandbox/current
systemctl start --no-block flash-sandbox.service flash-sandbox-scheduler.service
echo 'Updated release published; services starting'
