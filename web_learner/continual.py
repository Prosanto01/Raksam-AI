from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

MANIFEST = Path('data/web_knowledge/manifest.jsonl')
CANDIDATE = Path('brain/checkpoints/candidate.pt')
ACTIVE = Path('brain/checkpoints/latest.pt')
BACKUP = Path('brain/checkpoints/previous.pt')
STATE = Path('data/self_learning/state.json')


def new_document_count(manifest: str = str(MANIFEST)) -> int:
    p = Path(manifest)
    if not p.exists(): return 0
    return sum(1 for _ in p.open('r', encoding='utf-8'))


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout + '\n' + proc.stderr)[-12000:]


def _load_state() -> dict:
    if STATE.exists():
        try: return json.loads(STATE.read_text(encoding='utf-8'))
        except Exception: pass
    return {'cycles': 0, 'promotions': 0}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def _checkpoint_training_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        ck = __import__("torch").load(path, map_location="cpu", weights_only=False)
        return dict(ck.get("training", {}))
    except Exception:
        return {}

def promote_candidate() -> bool:
    if not CANDIDATE.exists(): raise FileNotFoundError(CANDIDATE)
    candidate_metadata = _checkpoint_training_metadata(CANDIDATE)
    active_metadata = _checkpoint_training_metadata(ACTIVE)
    candidate_signature = candidate_metadata.get("validation_signature")
    active_signature = active_metadata.get("validation_signature")
    # Losses from different validation sets cannot be compared. Older
    # checkpoints without a signature stay protected until manually evaluated.
    if ACTIVE.exists() and (not candidate_signature or candidate_signature != active_signature):
        return False
    candidate_loss = float(candidate_metadata.get("val_loss", float("inf")))
    active_loss = float(active_metadata.get("val_loss", float("inf")))
    if ACTIVE.exists() and candidate_loss >= active_loss:
        return False
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    if ACTIVE.exists(): ACTIVE.replace(BACKUP)
    CANDIDATE.replace(ACTIVE)
    return True


def run_learning(config_path: str = 'config/web_learning.yaml') -> dict:
    import yaml
    from web_learner.crawler import PublicWebLearner
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8')) or {}
    if not cfg.get('enabled', False): return {'status': 'disabled'}

    learner = PublicWebLearner(config_path)
    before = len(learner.seen)
    crawl = learner.crawl()
    added = len(learner.seen) - before
    result = {'status': 'ingested', **crawl, 'new_documents': added}
    state = _load_state(); state['cycles'] = int(state.get('cycles', 0)) + 1

    auto_train = bool(cfg.get('auto_train', False))
    threshold = int(cfg.get('min_new_documents', 5))
    if auto_train and (added >= threshold or not ACTIVE.exists()):
        epochs = str(int(cfg.get('auto_train_epochs', 1)))
        # This cycle has already crawled, so skip train.py's normal collection
        # preflight and avoid a duplicate pass over the configured sources.
        cmd = [sys.executable, 'brain/train.py', '--candidate', '--resume', '--epochs', epochs, '--skip-web-collection']
        rc, log = _run(cmd)
        result['training_returncode'] = rc; result['training_log'] = log
        if rc == 0 and CANDIDATE.exists():
            # train.py writes a candidate only when validation improves during training.
            # Promotion is additionally controlled by the explicit auto_promote switch.
            if bool(cfg.get('auto_promote', True)):
                try:
                    promoted = promote_candidate()
                    if promoted:
                        state['promotions'] = int(state.get('promotions', 0)) + 1
                    result['promoted'] = promoted
                except Exception as exc:
                    result['promoted'] = False; result['promotion_error'] = str(exc)
            else:
                result['promoted'] = False
    state['last_cycle'] = result
    _save_state(state)
    return result
