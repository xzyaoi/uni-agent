"""Fleet sizing and recovery checks that never contact AWS or SSH hosts."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "agentenv_multinode_deploy", Path(__file__).resolve().parents[2] / "cscs/agentenv_multinode/deploy.py"
)
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def arguments(tmp_path, count=2):
    return SimpleNamespace(
        count=count, name="throughput-2x", region="us-east-1", ami=deploy.AMI,
        state=tmp_path, apply=True, subnet=None, allow_cidr=["192.0.2.1/32"],
    )


def host(index, tagged=True):
    return {
        "InstanceId": f"i-{index}", "InstanceType": deploy.TYPE,
        "PrivateIpAddress": f"10.0.0.{index}", "PublicDnsName": f"host-{index}.example.com",
        "Tags": [{"Key": "AgentEnvExperiment", "Value": "throughput-2x"}] if tagged else [],
    }


@pytest.mark.parametrize(
    ("count", "existing", "quota", "additional", "quota_ok"),
    [(2, 0, 192, 192, True), (4, 0, 192, 384, False), (2, 2, 192, 0, True),
     (2, 3, 192, 0, False)],
)
def test_inventory_counts_requested_vcpus(monkeypatch, tmp_path, count, existing, quota, additional, quota_ok):
    def aws(args, service, command, *flags):
        return {
            "describe-instances": {"Reservations": [{"Instances": [host(i) for i in range(existing)]}]},
            "describe-instance-types": {"InstanceTypes": [
                {"InstanceType": deploy.TYPE, "VCpuInfo": {"DefaultVCpus": 96}}
            ]},
            "get-service-quota": {"Quota": {"Value": quota}},
            "describe-security-groups": {"SecurityGroups": [{"IpPermissions": []}]},
        }[command]

    monkeypatch.setattr(deploy, "aws", aws)
    result = deploy.inventory(arguments(tmp_path, count))
    assert result["additional_vcpus"] == additional
    assert result["quota_ok"] is quota_ok


def test_inventory_counts_stopping_hosts_against_quota(monkeypatch, tmp_path):
    stopping = host(1, tagged=False)
    stopping["State"] = {"Name": "stopping"}

    def aws(args, service, command, *flags):
        if command == "describe-instances":
            assert "Name=instance-state-name,Values=pending,running,stopping" in flags
            return {"Reservations": [{"Instances": [stopping]}]}
        return {
            "describe-instance-types": {"InstanceTypes": [
                {"InstanceType": deploy.TYPE, "VCpuInfo": {"DefaultVCpus": 96}}
            ]},
            "get-service-quota": {"Quota": {"Value": 192}},
            "describe-security-groups": {"SecurityGroups": [{"IpPermissions": []}]},
        }[command]

    monkeypatch.setattr(deploy, "aws", aws)
    result = deploy.inventory(arguments(tmp_path))
    assert result["used_standard_vcpus"] == 96
    assert result["additional_vcpus"] == 192
    assert result["quota_ok"] is False


def test_provision_requests_exactly_two_instances(monkeypatch, tmp_path):
    def aws(args, service, command, *flags):
        return {
            "describe-security-groups": {"SecurityGroups": [{"GroupId": "sg-existing"}]},
            "describe-subnets": {"Subnets": [{"SubnetId": "subnet-test", "AvailabilityZone": "us-east-1a"}]},
            "run-instances": {"Instances": [host(1), host(2)]},
        }.get(command, {})

    mocked_aws = Mock(side_effect=aws)
    monkeypatch.setattr(deploy, "aws", mocked_aws)
    state = {}
    deploy.provision(arguments(tmp_path), state, {
        "instances": [], "quota_ok": True, "source_cidrs": ["192.0.2.1/32"],
    })
    launch = next(call.args for call in mocked_aws.call_args_list if call.args[2] == "run-instances")
    assert launch[launch.index("--count") + 1] == "2"
    assert "--min-count" not in launch
    assert "--max-count" not in launch
    assert state["instance_ids"] == ["i-1", "i-2"]
    assert json.loads((tmp_path / "state.json").read_text())["client_token"] == state["client_token"]


@pytest.mark.parametrize(("existing", "quota_ok"), [(1, True), (3, True), (0, False)])
def test_provision_rejects_partial_fleet_or_insufficient_quota(monkeypatch, tmp_path, existing, quota_ok):
    mocked_aws = Mock()
    monkeypatch.setattr(deploy, "aws", mocked_aws)
    with pytest.raises(RuntimeError):
        deploy.provision(arguments(tmp_path), {}, {
            "instances": [host(i) for i in range(existing)], "quota_ok": quota_ok,
        })
    mocked_aws.assert_not_called()


def test_provision_recovers_existing_two_host_fleet(monkeypatch, tmp_path):
    mocked_aws = Mock()
    monkeypatch.setattr(deploy, "aws", mocked_aws)
    state = {}
    deploy.provision(arguments(tmp_path), state, {"instances": [host(2), host(1)]})
    assert state["instance_ids"] == ["i-1", "i-2"]
    mocked_aws.assert_not_called()


def invoke_main(monkeypatch, tmp_path, command="plan", count=2, *extra):
    monkeypatch.setattr(sys, "argv", [
        "deploy.py", command, "--state", str(tmp_path), "--name", "throughput-2x", "--count", str(count), *extra,
    ])
    deploy.main()


def empty_inventory(args):
    return {"instances": [], "quota_ok": True, "additional_vcpus": args.count * 96}


def test_count_is_persisted_in_plan_and_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(deploy, "inventory", empty_inventory)
    invoke_main(monkeypatch, tmp_path)
    assert json.loads((tmp_path / "state.json").read_text())["count"] == 2
    assert json.loads((tmp_path / "plan.json").read_text())["runtime_count"] == 2
    mocked_inventory = Mock()
    monkeypatch.setattr(deploy, "inventory", mocked_inventory)
    with pytest.raises(ValueError, match="State identity mismatch"):
        invoke_main(monkeypatch, tmp_path, count=4)
    mocked_inventory.assert_not_called()


def test_legacy_state_is_four_hosts(monkeypatch, tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({
        "name": "throughput-2x", "region": "us-east-1", "ami": deploy.AMI, "phase": "planned",
    }))
    monkeypatch.setattr(deploy, "inventory", empty_inventory)
    with pytest.raises(ValueError, match="State identity mismatch"):
        invoke_main(monkeypatch, tmp_path, count=2)
    invoke_main(monkeypatch, tmp_path, count=4)
    assert json.loads((tmp_path / "state.json").read_text())["count"] == 4


def test_default_count_remains_four(monkeypatch, tmp_path):
    monkeypatch.setattr(deploy, "inventory", empty_inventory)
    monkeypatch.setattr(sys, "argv", ["deploy.py", "plan", "--state", str(tmp_path)])
    deploy.main()
    assert json.loads((tmp_path / "state.json").read_text())["count"] == 4
    assert json.loads((tmp_path / "plan.json").read_text())["runtime_count"] == 4


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_count_fails_before_aws(monkeypatch, tmp_path, count):
    mocked_inventory = Mock()
    monkeypatch.setattr(deploy, "inventory", mocked_inventory)
    with pytest.raises(SystemExit) as error:
        invoke_main(monkeypatch, tmp_path, count=count)
    assert error.value.code == 2
    mocked_inventory.assert_not_called()


def test_inspection_rejects_incomplete_aws_response(monkeypatch, tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({
        "name": "throughput-2x", "region": "us-east-1", "ami": deploy.AMI, "count": 2,
        "phase": "booting_needs_runtime_inspection", "instance_ids": ["i-1", "i-2"],
    }))
    monkeypatch.setattr(deploy, "inventory", empty_inventory)
    monkeypatch.setattr(deploy, "aws", lambda *args: {"Reservations": [{"Instances": [host(1)]}]})
    mocked_ssh = Mock()
    monkeypatch.setattr(deploy, "ssh", mocked_ssh)
    with pytest.raises(ValueError, match="AWS instance response differs"):
        invoke_main(monkeypatch, tmp_path, "inspect")
    mocked_ssh.assert_not_called()


def test_inspect_and_configure_two_hosts(monkeypatch, tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({
        "name": "throughput-2x", "region": "us-east-1", "ami": deploy.AMI, "count": 2,
        "phase": "booting_needs_runtime_inspection", "instance_ids": ["i-1", "i-2"],
    }))
    monkeypatch.setattr(deploy, "inventory", empty_inventory)
    monkeypatch.setattr(deploy, "aws", lambda *args: {"Reservations": [{"Instances": [host(2), host(1)]}]})
    digest = "a" * 64
    report = {
        "runtime_sha256": digest, "source_commit": {"stdout": deploy.PINNED_COMMIT},
        "kvm": True, "health": {"returncode": 0}, "runtime_api_addr": "0.0.0.0:8000",
        "recorder_active": {"returncode": 3},
    }
    mocked_ssh = Mock(return_value=json.dumps(report))
    monkeypatch.setattr(deploy, "ssh", mocked_ssh)
    invoke_main(monkeypatch, tmp_path, "inspect")
    assert mocked_ssh.call_count == 2
    provenance = tmp_path / "reviewed.json"
    provenance.write_text(json.dumps({
        "commit": deploy.PINNED_COMMIT, "runtime_sha256": digest, "heartbeat_supported": True,
        "shared_storage_verified": True, "runtime_api_addr": "0.0.0.0:8000", "recorder_active": False,
    }))
    mocked_ssh.reset_mock()
    invoke_main(monkeypatch, tmp_path, "configure", 2, "--apply", "--compatibility", str(provenance))
    configured = [call.args[3] for call in mocked_ssh.call_args_list if call.args[2] == deploy.CONFIGURE]
    assert [payload["node_id"] for payload in configured] == ["runtime-1", "runtime-2"]
    config = json.loads((tmp_path / "control-plane.json").read_text())
    assert len(config["scheduler"]["nodes"]) == 2
    assert json.loads((tmp_path / "state.json").read_text())["configured_ids"] == ["i-1", "i-2"]
