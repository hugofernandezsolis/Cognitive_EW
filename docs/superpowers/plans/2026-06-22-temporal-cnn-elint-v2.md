# Temporal CNN ELINT v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Replace M2's current LPI-only anchor shortcut with an ELINT v2 path that passes only when `macro_acc_type`, `macro_acc_mode`, `macro_acc_threat`, and `lpi_accuracy` are all `>= 0.96`, and `latency_p99_ms < 1.0`.

**Architecture:** Keep the existing synthetic PDW/ELINT approach required by the proposal. Add a richer v2 PDW feature set, a lightweight three-head temporal CNN, strict metrics, and experiment profiles that use the stricter anchor. Preserve the current v1 model/config path for compatibility with M4 robustness tests and existing documentation.

**Tech Stack:** Python 3.12, PyTorch, NumPy, PyYAML, pytest.

---

## File Structure

Files to modify:

- `src/cog_ew/data/pdw_dataset.py`: add `PDWConfig.feature_set` and v2 feature channels.
- `tests/data/test_pdw_dataset.py`: cover base/v2 shapes, labels, and deterministic v2 features.
- `src/cog_ew/temporal_cnn_elint/model.py`: add `TemporalCNNV2Config` and `TemporalCNNV2`.
- `tests/temporal_cnn_elint/test_model.py`: cover v1 compatibility and v2 three-head outputs.
- `src/cog_ew/temporal_cnn_elint/metrics.py`: add strict M2 score/pass helpers and tighten latency profiling.
- `tests/temporal_cnn_elint/test_metrics.py`: cover strict score/pass behavior.
- `src/cog_ew/temporal_cnn_elint/train.py`: add architecture selection, v2 model construction, threat loss, and threat metrics.
- `tests/temporal_cnn_elint/test_train.py`: cover v2 config parsing and v2 smoke training outputs.
- `configs/temporal_cnn_elint/train_v2.yaml`: new full M2-v2 training config.
- `configs/experiments/quick.yaml`: route quick ELINT anchor through v2 with the quick epoch override.
- `configs/experiments/full.yaml`: route full ELINT anchor through v2.
- `src/cog_ew/experiments/anchors.py`: use strict M2 achieved score and pass condition.
- `tests/experiments/test_anchors.py`: cover strict ELINT anchor result.
- `tests/experiments/test_report.py`: update pass condition expectation for strict ELINT.
- `docs/ROADMAP.md`: mark M2 as v2/strict once implementation is verified.
- `changes.md`: record what changed and why.

---

## Task 1: Add PDW v2 Feature Set

**Objective:** Keep current 10-channel v1 PDW windows unchanged, and add an explicit 18-channel v2 representation that encodes separability needed by the proposal.

### 1.1 Write failing dataset tests

Edit `tests/data/test_pdw_dataset.py`:

```python
def test_dataset_v2_feature_shape():
    ds = PDWSyntheticDataset(_config(feature_set="v2"))

    pdw, type_idx, mode_idx, threat_idx = ds[0]

    assert pdw.shape == (18, 64)
    assert pdw.dtype == torch.float32
    assert type_idx == 6
    assert mode_idx == 0
    assert threat_idx == 0


def test_dataset_v2_features_are_deterministic():
    a = PDWSyntheticDataset(_config(feature_set="v2", seed=123))[0][0]
    b = PDWSyntheticDataset(_config(feature_set="v2", seed=123))[0][0]

    assert torch.allclose(a, b)


def test_dataset_rejects_unknown_feature_set():
    try:
        PDWSyntheticDataset(_config(feature_set="unknown"))
    except ValueError as exc:
        assert "feature_set" in str(exc)
    else:
        raise AssertionError("PDWConfig accepted an unknown feature_set")
```

Run:

```bash
.venv/bin/pytest tests/data/test_pdw_dataset.py
```

Expected result: the new v2 tests fail because `PDWConfig` does not accept `feature_set`.

### 1.2 Implement `feature_set`

Edit `src/cog_ew/data/pdw_dataset.py`.

Add the config field:

```python
feature_set: str = "base"
```

Validate in `PDWSyntheticDataset.__init__`:

```python
if config.feature_set not in {"base", "v2"}:
    raise ValueError("feature_set must be 'base' or 'v2'")
```

Refactor feature assembly into helpers:

```python
def _base_features(self, pulses: np.ndarray, mode_name: str) -> np.ndarray:
    rf = _normalize(pulses["rf_mhz"], center=10_000.0, scale=1_000.0)
    pw = _normalize(pulses["pw_us"], center=20.0, scale=20.0)
    pa = _normalize(pulses["pa_db"], center=-40.0, scale=20.0)
    aoa = _normalize(pulses["aoa_deg"], center=0.0, scale=90.0)
    pri = _normalize(pulses["pri_us"], center=500.0, scale=500.0)
    cont = np.stack([rf, pw, pa, aoa, pri], axis=1)
    intra = _one_hot_intra_pulse(mode_name, len(pulses))
    return np.concatenate([cont, intra], axis=1).astype(np.float32)
```

Add v2 channels after the 10 base channels:

```python
def _v2_features(self, pulses: np.ndarray, mode_name: str) -> np.ndarray:
    base = self._base_features(pulses, mode_name)
    rf = base[:, 0]
    pw = base[:, 1]
    pri = base[:, 4]
    delta_rf = _prepend_zero(np.diff(rf))
    delta_pri = _prepend_zero(np.diff(pri))
    rolling_pri_std = _rolling_std(pri, width=8)
    rolling_rf_std = _rolling_std(rf, width=8)
    rolling_pw_mean = _rolling_mean(pw, width=8)
    pulse_progression = _pulse_progression(len(pulses))
    lpi_hint = _lpi_hint(pulses)
    freq_hopping_hint = _freq_hopping_hint(rf)
    extra = np.stack(
        [
            delta_rf,
            delta_pri,
            rolling_pri_std,
            rolling_rf_std,
            rolling_pw_mean,
            pulse_progression,
            lpi_hint,
            freq_hopping_hint,
        ],
        axis=1,
    )
    return np.concatenate([base, extra], axis=1).astype(np.float32)
```

Use pure NumPy helpers:

```python
def _prepend_zero(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=np.float32), values.astype(np.float32)])


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i + 1 - width)
        out[i] = float(np.mean(values[start : i + 1]))
    return out


def _rolling_std(values: np.ndarray, width: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i + 1 - width)
        out[i] = float(np.std(values[start : i + 1]))
    return out
```

Mode/threat hints should be weak, proposal-aligned labels rather than target leakage from sample IDs:

```python
def _pulse_progression(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def _lpi_hint(pulses: np.ndarray) -> np.ndarray:
    bandwidth = float(np.std(pulses["rf_mhz"]))
    duty_hint = float(np.mean(pulses["pw_us"]) / max(np.mean(pulses["pri_us"]), 1e-6))
    raw = 0.5 * np.tanh(bandwidth / 120.0) + 0.5 * np.tanh(duty_hint * 20.0)
    return np.full(len(pulses), np.clip(raw, 0.0, 1.0), dtype=np.float32)


def _freq_hopping_hint(rf: np.ndarray) -> np.ndarray:
    return np.tanh(np.abs(_prepend_zero(np.diff(rf))) * 4.0).astype(np.float32)
```

Use v2 only when requested:

```python
features = (
    self._v2_features(pulses, mode_name)
    if self.config.feature_set == "v2"
    else self._base_features(pulses, mode_name)
)
```

Run:

```bash
.venv/bin/pytest tests/data/test_pdw_dataset.py tests/gan_signals/test_robustness.py
```

Expected result: pass. The robustness test confirms v1 compatibility remains intact.

Commit:

```bash
git add src/cog_ew/data/pdw_dataset.py tests/data/test_pdw_dataset.py
git commit -m "feat: add v2 PDW feature set"
```

---

## Task 2: Add Three-Head TemporalCNNV2

**Objective:** Preserve `TemporalCNN` v1 API, and add a v2 model that predicts type, mode, and threat directly.

### 2.1 Write failing model tests

Edit `tests/temporal_cnn_elint/test_model.py`:

```python
from cog_ew.temporal_cnn_elint.model import TemporalCNNV2, TemporalCNNV2Config


def test_v2_forward_output_shapes():
    model = TemporalCNNV2(TemporalCNNV2Config())
    x = torch.randn(4, 18, 64)

    type_logits, mode_logits, threat_logits = model(x)

    assert type_logits.shape == (4, 8)
    assert mode_logits.shape == (4, 4)
    assert threat_logits.shape == (4, 3)


def test_v2_predict_shapes_and_dtypes():
    model = TemporalCNNV2(TemporalCNNV2Config())
    x = torch.randn(5, 18, 64)

    type_pred, mode_pred, threat_pred = model.predict(x)

    assert type_pred.shape == (5,)
    assert mode_pred.shape == (5,)
    assert threat_pred.shape == (5,)
    assert type_pred.dtype == torch.long
    assert mode_pred.dtype == torch.long
    assert threat_pred.dtype == torch.long


def test_v2_param_count_stays_lightweight():
    model = TemporalCNNV2(TemporalCNNV2Config())
    n_params = sum(p.numel() for p in model.parameters())

    assert 40_000 < n_params < 250_000
```

Run:

```bash
.venv/bin/pytest tests/temporal_cnn_elint/test_model.py
```

Expected result: import failure for `TemporalCNNV2`.

### 2.2 Implement v2 config and model

Edit `src/cog_ew/temporal_cnn_elint/model.py`.

Add config:

```python
@dataclass(frozen=True)
class TemporalCNNV2Config:
    in_channels: int = 18
    n_types: int = 8
    n_modes: int = 4
    n_threats: int = 3
    hidden: int = 64
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.05
```

Add a depthwise residual block:

```python
class _DepthwiseTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))
```

Add `TemporalCNNV2`:

```python
class TemporalCNNV2(nn.Module):
    def __init__(self, config: TemporalCNNV2Config) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.in_channels, config.hidden, kernel_size=1),
            nn.BatchNorm1d(config.hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                _DepthwiseTCNBlock(config.hidden, config.kernel_size, dilation, config.dropout)
                for dilation in config.dilations
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.type_head = nn.Linear(config.hidden, config.n_types)
        self.mode_head = nn.Linear(config.hidden, config.n_modes)
        self.threat_head = nn.Linear(config.hidden, config.n_threats)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.blocks(h)
        z = self.pool(h).squeeze(-1)
        return self.type_head(z), self.mode_head(z), self.threat_head(z)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        type_logits, mode_logits, threat_logits = self(x)
        return type_logits.argmax(1), mode_logits.argmax(1), threat_logits.argmax(1)
```

Run:

```bash
.venv/bin/pytest tests/temporal_cnn_elint/test_model.py
```

Expected result: pass.

Commit:

```bash
git add src/cog_ew/temporal_cnn_elint/model.py tests/temporal_cnn_elint/test_model.py
git commit -m "feat: add three-head TemporalCNN v2"
```

---

## Task 3: Add Strict M2 Metrics

**Objective:** Make M2 scoring reflect the proposal: the anchor can no longer pass from LPI alone.

### 3.1 Write failing metric tests

Edit `tests/temporal_cnn_elint/test_metrics.py`:

```python
from cog_ew.temporal_cnn_elint.metrics import strict_elint_passed, strict_elint_score


def test_strict_elint_score_uses_worst_required_metric():
    metrics = {
        "macro_acc_type": 1.0,
        "macro_acc_mode": 0.97,
        "macro_acc_threat": 0.98,
        "lpi_accuracy": 0.99,
        "latency_p99_ms": 0.7,
    }

    assert strict_elint_score(metrics) == 0.97


def test_strict_elint_score_zero_when_latency_fails():
    metrics = {
        "macro_acc_type": 1.0,
        "macro_acc_mode": 1.0,
        "macro_acc_threat": 1.0,
        "lpi_accuracy": 1.0,
        "latency_p99_ms": 1.01,
    }

    assert strict_elint_score(metrics) == 0.0


def test_strict_elint_passed_requires_all_metrics_and_latency():
    passing = {
        "macro_acc_type": 0.96,
        "macro_acc_mode": 0.97,
        "macro_acc_threat": 0.98,
        "lpi_accuracy": 0.99,
        "latency_p99_ms": 0.99,
    }
    failing = dict(passing, macro_acc_mode=0.95)

    assert strict_elint_passed(passing, target=0.96, latency_p99_ms=1.0)
    assert not strict_elint_passed(failing, target=0.96, latency_p99_ms=1.0)
```

Run:

```bash
.venv/bin/pytest tests/temporal_cnn_elint/test_metrics.py
```

Expected result: import failure for strict helpers.

### 3.2 Implement strict helpers

Edit `src/cog_ew/temporal_cnn_elint/metrics.py`:

```python
STRICT_ELINT_KEYS = ("macro_acc_type", "macro_acc_mode", "macro_acc_threat", "lpi_accuracy")


def strict_elint_score(metrics: Mapping[str, float], latency_p99_ms: float = 1.0) -> float:
    if float(metrics["latency_p99_ms"]) >= latency_p99_ms:
        return 0.0
    return min(float(metrics[key]) for key in STRICT_ELINT_KEYS)


def strict_elint_passed(metrics: Mapping[str, float], target: float = 0.96, latency_p99_ms: float = 1.0) -> bool:
    return strict_elint_score(metrics, latency_p99_ms=latency_p99_ms) >= target
```

Update `profile_latency` to use inference mode:

```python
with torch.inference_mode():
    for _ in range(warmup):
        model(sample)
    for _ in range(runs):
        start = time.perf_counter()
        model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize(sample.device)
        timings.append((time.perf_counter() - start) * 1000.0)
```

Run:

```bash
.venv/bin/pytest tests/temporal_cnn_elint/test_metrics.py
```

Expected result: pass.

Commit:

```bash
git add src/cog_ew/temporal_cnn_elint/metrics.py tests/temporal_cnn_elint/test_metrics.py
git commit -m "feat: add strict ELINT anchor metrics"
```

---

## Task 4: Train v2 With Threat Head

**Objective:** Allow `train.py` to select v1 or v2, train three heads for v2, and emit strict-ready metrics.

### 4.1 Write failing train tests

Edit `tests/temporal_cnn_elint/test_train.py`.

Add imports:

```python
from cog_ew.temporal_cnn_elint.model import TemporalCNNV2Config
```

Add tests:

```python
def test_train_config_v2_yaml_parses():
    config = TrainConfig.from_yaml("configs/temporal_cnn_elint/train_v2.yaml")

    assert config.architecture == "v2"
    assert isinstance(config.model, TemporalCNNV2Config)
    assert config.data.feature_set == "v2"
    assert len(config.loss_weights) == 3


def test_train_v2_smoke_outputs_threat_metrics(tmp_path):
    data = PDWConfig(
        library_path="configs/temporal_cnn_elint/emitters.yaml",
        emitters=("LPI-FMCW", "Pulse-Doppler"),
        modes=("search", "track"),
        window=64,
        n_pulses=128,
        n_trains=4,
        seed=1,
        feature_set="v2",
    )
    model = TemporalCNNV2Config(hidden=16, dilations=(1,), dropout=0.0)
    config = TrainConfig(
        data=data,
        model=model,
        architecture="v2",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        out_dir=str(tmp_path),
        latency_runs=2,
        latency_warmup=1,
    )

    result = train(config)

    assert "macro_acc_threat" in result.metrics
    assert "confusion_threat" in result.metrics
    assert result.metrics["macro_acc_type"] >= 0.0
    assert result.metrics["macro_acc_mode"] >= 0.0
    assert result.metrics["macro_acc_threat"] >= 0.0
```

Run:

```bash
.venv/bin/pytest tests/temporal_cnn_elint/test_train.py
```

Expected result: failures for missing `train_v2.yaml`, `architecture`, and v2 training support.

### 4.2 Extend train config

Edit `src/cog_ew/temporal_cnn_elint/train.py`.

Imports:

```python
from cog_ew.temporal_cnn_elint.model import TemporalCNN, TemporalCNNConfig, TemporalCNNV2, TemporalCNNV2Config
from cog_ew.temporal_cnn_elint.metrics import confusion_matrix, lpi_accuracy, macro_accuracy, profile_latency, strict_elint_score
```

Update dataclass:

```python
ModelConfig = TemporalCNNConfig | TemporalCNNV2Config


@dataclass(frozen=True)
class TrainConfig:
    data: PDWConfig
    model: ModelConfig
    architecture: str = "v1"
    epochs: int = 8
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0001
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 0
    device: str | None = None
    out_dir: str = "runs/temporal_cnn_elint"
    latency_runs: int = 100
    latency_warmup: int = 10
    loss_weights: tuple[float, ...] = (1.0, 1.0)
```

In `from_yaml`, parse architecture before model:

```python
architecture = raw.get("architecture", "v1")
if architecture == "v2":
    model = TemporalCNNV2Config(**raw.get("model", {}))
elif architecture == "v1":
    model = TemporalCNNConfig(**raw.get("model", {}))
else:
    raise ValueError("architecture must be 'v1' or 'v2'")
```

Preserve existing tuple parsing for `split` and `loss_weights`.

### 4.3 Add model factory and generic prediction collection

Add:

```python
def _build_model(config: TrainConfig) -> nn.Module:
    if config.architecture == "v2":
        if not isinstance(config.model, TemporalCNNV2Config):
            raise TypeError("v2 architecture requires TemporalCNNV2Config")
        return TemporalCNNV2(config.model)
    if not isinstance(config.model, TemporalCNNConfig):
        raise TypeError("v1 architecture requires TemporalCNNConfig")
    return TemporalCNN(config.model)
```

Update `_collect_preds` so v1 still derives threat from mode, while v2 uses the third head:

```python
def _collect_preds(model: nn.Module, loader: DataLoader, device: torch.device, architecture: str) -> dict[str, np.ndarray]:
    y_type: list[np.ndarray] = []
    y_mode: list[np.ndarray] = []
    y_threat: list[np.ndarray] = []
    p_type: list[np.ndarray] = []
    p_mode: list[np.ndarray] = []
    p_threat: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, yt, ym, yth in loader:
            x = x.to(device)
            yt = yt.to(device)
            ym = ym.to(device)
            yth = yth.to(device)
            if architecture == "v2":
                lt, lm, lth = model(x)
                pred_threat = lth.argmax(1)
            else:
                lt, lm = model(x)
                pred_threat = model.predict(x)[2].to(device)
            y_type.append(yt.cpu().numpy())
            y_mode.append(ym.cpu().numpy())
            y_threat.append(yth.cpu().numpy())
            p_type.append(lt.argmax(1).cpu().numpy())
            p_mode.append(lm.argmax(1).cpu().numpy())
            p_threat.append(pred_threat.cpu().numpy())
    return {
        "y_type": np.concatenate(y_type),
        "y_mode": np.concatenate(y_mode),
        "y_threat": np.concatenate(y_threat),
        "p_type": np.concatenate(p_type),
        "p_mode": np.concatenate(p_mode),
        "p_threat": np.concatenate(p_threat),
    }
```

### 4.4 Update training loop and metrics

Use `_build_model(config)`.

For v2 batches:

```python
if config.architecture == "v2":
    type_logits, mode_logits, threat_logits = model(x)
    loss = (
        config.loss_weights[0] * ce(type_logits, type_y)
        + config.loss_weights[1] * ce(mode_logits, mode_y)
        + config.loss_weights[2] * ce(threat_logits, threat_y)
    )
else:
    type_logits, mode_logits = model(x)
    loss = config.loss_weights[0] * ce(type_logits, type_y) + config.loss_weights[1] * ce(mode_logits, mode_y)
```

Build metrics:

```python
metrics = {
    "macro_acc_type": macro_accuracy(preds["y_type"], preds["p_type"], n_classes=dataset.n_types),
    "macro_acc_mode": macro_accuracy(preds["y_mode"], preds["p_mode"], n_classes=dataset.n_modes),
    "macro_acc_threat": macro_accuracy(preds["y_threat"], preds["p_threat"], n_classes=3),
    "lpi_accuracy": lpi_accuracy(preds["y_type"], preds["p_type"], dataset.type_names, lpi_prefix="LPI"),
    "confusion_type": confusion_matrix(preds["y_type"], preds["p_type"], n_classes=dataset.n_types).tolist(),
    "confusion_mode": confusion_matrix(preds["y_mode"], preds["p_mode"], n_classes=dataset.n_modes).tolist(),
    "confusion_threat": confusion_matrix(preds["y_threat"], preds["p_threat"], n_classes=3).tolist(),
}
```

After latency:

```python
metrics["strict_elint_score"] = strict_elint_score(metrics)
```

Run:

```bash
.venv/bin/pytest tests/temporal_cnn_elint/test_train.py tests/temporal_cnn_elint/test_metrics.py
```

Expected result: train tests still fail only because `train_v2.yaml` has not been added.

Commit after Task 5 because this task depends on the new config file.

---

## Task 5: Add M2-v2 Training Config

**Objective:** Provide a reproducible full-profile config for the stricter M2.

### 5.1 Add config file

Create `configs/temporal_cnn_elint/train_v2.yaml`:

```yaml
architecture: v2
data:
  library_path: configs/temporal_cnn_elint/emitters.yaml
  emitters:
    - LPI-FMCW
    - LPI-PhaseCoded
    - Pulse-Doppler
    - Search-Radar
    - Track-Radar
    - Fire-Control
    - Naval-Air-Search
    - SAM-Engagement
  modes:
    - search
    - tws
    - track
    - missile_guidance
  window: 64
  n_pulses: 512
  n_trains: 96
  normalize: true
  noise_std: 0.015
  drop_prob: 0.01
  spurious_prob: 0.01
  seed: 0
  feature_set: v2
model:
  in_channels: 18
  n_types: 8
  n_modes: 4
  n_threats: 4
  hidden: 32
  kernel_size: 5
  dilations:
    - 1
    - 2
    - 4
  dropout: 0.05
epochs: 24
batch_size: 128
lr: 0.001
weight_decay: 0.0001
split:
  - 0.7
  - 0.15
  - 0.15
seed: 0
device: null
out_dir: runs/temporal_cnn_elint_v2
latency_runs: 100
latency_warmup: 10
loss_weights:
  - 1.0
  - 1.1
  - 1.0
```

Run:

```bash
.venv/bin/pytest tests/temporal_cnn_elint/test_train.py
```

Expected result: pass.

Commit Tasks 4 and 5 together:

```bash
git add src/cog_ew/temporal_cnn_elint/train.py tests/temporal_cnn_elint/test_train.py configs/temporal_cnn_elint/train_v2.yaml
git commit -m "feat: train ELINT v2 with threat head"
```

---

## Task 6: Enforce Strict ELINT Anchor

**Objective:** Make experiment anchors report M2 achieved score as the strict minimum, with latency gating.

### 6.1 Write failing anchor/report tests

Edit `tests/experiments/test_anchors.py`:

```python
def test_run_elint_anchor_uses_strict_score(tmp_path):
    profile = ExperimentProfile.quick()
    result = run_elint_anchor(profile, tmp_path)
    metrics_path = Path(result.run_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())

    expected = 0.0 if metrics["latency_p99_ms"] >= 1.0 else min(
        metrics["macro_acc_type"],
        metrics["macro_acc_mode"],
        metrics["macro_acc_threat"],
        metrics["lpi_accuracy"],
    )

    assert result.achieved == expected
    assert result.passed == (expected >= 0.96)
```

Edit `tests/experiments/test_report.py`:

```python
def test_run_anchors_elint_uses_strict_score(tmp_path):
    profile = ExperimentProfile.quick()
    report = run_anchors(("elint",), profile, tmp_path)
    elint = report["anchors"]["elint"]
    metrics = json.loads((Path(elint["run_dir"]) / "metrics.json").read_text())
    expected = 0.0 if metrics["latency_p99_ms"] >= 1.0 else min(
        metrics["macro_acc_type"],
        metrics["macro_acc_mode"],
        metrics["macro_acc_threat"],
        metrics["lpi_accuracy"],
    )

    assert elint["achieved"] == expected
    assert elint["passed"] == (expected >= elint["target"])
```

Run:

```bash
.venv/bin/pytest tests/experiments/test_anchors.py tests/experiments/test_report.py
```

Expected result: failure because current anchor uses `lpi_accuracy`.

### 6.2 Update anchor implementation

Edit `src/cog_ew/experiments/anchors.py`.

Import strict helper:

```python
from cog_ew.temporal_cnn_elint.metrics import strict_elint_passed, strict_elint_score
```

Update `run_elint_anchor`:

```python
metrics = result.metrics
achieved = strict_elint_score(metrics)
passed = strict_elint_passed(metrics, target=_TARGETS["elint"])
return AnchorResult(
    name="elint",
    target=_TARGETS["elint"],
    achieved=achieved,
    baseline=None,
    passed=passed,
    run_dir=str(run_dir),
)
```

### 6.3 Route experiment profiles to v2

Edit both experiment configs:

`configs/experiments/quick.yaml`:

```yaml
elint_config: configs/temporal_cnn_elint/train_v2.yaml
elint_epochs: 2
```

`configs/experiments/full.yaml`:

```yaml
elint_config: configs/temporal_cnn_elint/train_v2.yaml
elint_epochs: null
```

Run:

```bash
.venv/bin/pytest tests/experiments/test_anchors.py tests/experiments/test_report.py
```

Expected result: pass.

Commit:

```bash
git add src/cog_ew/experiments/anchors.py tests/experiments/test_anchors.py tests/experiments/test_report.py configs/experiments/quick.yaml configs/experiments/full.yaml
git commit -m "fix: enforce strict ELINT anchor"
```

---

## Task 7: Verify Full M2-v2 Behavior

**Objective:** Prove the implementation meets the proposal-oriented target before updating docs.

### 7.1 Run focused test suite

Run:

```bash
.venv/bin/pytest \
  tests/data/test_pdw_dataset.py \
  tests/temporal_cnn_elint/test_model.py \
  tests/temporal_cnn_elint/test_metrics.py \
  tests/temporal_cnn_elint/test_train.py \
  tests/experiments/test_anchors.py \
  tests/experiments/test_report.py
```

Expected result: all selected tests pass.

### 7.2 Run full unit suite

Run:

```bash
.venv/bin/pytest
```

Expected result: all tests pass.

### 7.3 Run quick ELINT anchor

Run:

```bash
.venv/bin/python -m cog_ew.experiments.run_anchors --profile quick --anchors elint --out-dir runs/anchors_quick_m2_v2
```

Expected result: command exits `0`; `runs/anchors_quick_m2_v2/elint/metrics.json` contains `macro_acc_threat` and `strict_elint_score`.

### 7.4 Run full ELINT anchor

Run:

```bash
.venv/bin/python -m cog_ew.experiments.run_anchors --profile full --anchors elint --out-dir runs/anchors_full_m2_v2
```

Expected result: command exits `0`.

Inspect:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path("runs/anchors_full_m2_v2/elint/metrics.json").read_text())
print({
    "macro_acc_type": metrics["macro_acc_type"],
    "macro_acc_mode": metrics["macro_acc_mode"],
    "macro_acc_threat": metrics["macro_acc_threat"],
    "lpi_accuracy": metrics["lpi_accuracy"],
    "latency_p99_ms": metrics["latency_p99_ms"],
    "strict_elint_score": metrics["strict_elint_score"],
})
PY
```

Required values:

- `macro_acc_type >= 0.96`
- `macro_acc_mode >= 0.96`
- `macro_acc_threat >= 0.96`
- `lpi_accuracy >= 0.96`
- `latency_p99_ms < 1.0`
- `strict_elint_score >= 0.96`

If latency fails while accuracy passes on the target GPU, reduce `hidden` below `32` or remove another dilation in `train_v2.yaml`, rerun Tasks 5-7, and keep the highest-capacity config that satisfies `latency_p99_ms < 1.0`.

Commit only after full verification succeeds:

```bash
git add configs/temporal_cnn_elint/train_v2.yaml
git commit -m "tune: satisfy strict ELINT latency target"
```

Skip this commit if no tuning change was needed.

---

## Task 8: Update Roadmap and Changes

**Objective:** Make documentation match the implemented M2-v2 behavior and explain why it changed.

### 8.1 Update roadmap

Edit `docs/ROADMAP.md` so Modelo 2 no longer reads as just "Completo, mergeado". Use:

```markdown
| **Modelo 2** — Temporal CNN ELINT v2 | `src/cog_ew/temporal_cnn_elint/` | ✅ Estricto: type/mode/threat/LPI >= 0.96 y p99 < 1 ms |
```

Add a short note in the M2 section:

```markdown
- M2-v2 sustituye el criterio anterior basado en `lpi_accuracy` por una métrica estricta que exige precisión macro por tipo, modo y amenaza, además de LPI y latencia p99.
```

### 8.2 Update changes log

Edit `changes.md` and add an entry:

```markdown
## 2026-06-22 - M2-v2 ELINT estricto

- Añadido `feature_set: v2` al dataset PDW para exponer señales temporales derivadas sin romper el flujo v1 usado por otros modelos.
- Añadido `TemporalCNNV2`, con cabezales explícitos de tipo, modo y amenaza.
- Actualizado el entrenamiento de M2 para emitir `macro_acc_threat`, matrices de confusión por amenaza y `strict_elint_score`.
- Actualizada el ancla ELINT para que ya no pueda pasar usando solo `lpi_accuracy`; ahora exige type/mode/threat/LPI >= 0.96 y p99 < 1 ms.
- Actualizados los perfiles de experimento para ejecutar M2-v2, priorizando el cumplimiento literal de la propuesta.
```

Run:

```bash
.venv/bin/pytest
```

Expected result: all tests pass.

Commit:

```bash
git add docs/ROADMAP.md changes.md
git commit -m "docs: document strict ELINT v2"
```

---

## Final Verification

Run:

```bash
git status --short
.venv/bin/pytest
.venv/bin/python -m cog_ew.experiments.run_anchors --profile full --anchors elint --out-dir runs/anchors_full_m2_v2
```

Expected final state:

- Git status contains only intentional untracked run artifacts and unrelated pre-existing files.
- Unit tests pass.
- Full ELINT anchor passes under strict scoring.
- `runs/anchors_full_m2_v2/anchors_report.json` reports `"passed": true` for `elint`.
- `runs/anchors_full_m2_v2/elint/metrics.json` contains `macro_acc_type`, `macro_acc_mode`, `macro_acc_threat`, `lpi_accuracy`, `latency_p99_ms`, and `strict_elint_score`.

---

## Rollback Plan

If v2 fails the strict target after tuning:

1. Leave v1 files and configs intact.
2. Revert only commits from this plan in reverse order.
3. Keep `docs/superpowers/specs/2026-06-22-temporal-cnn-elint-v2-design.md` as the design record.
4. Record the failed full-anchor metrics in `changes.md` before attempting a different M2 approach.

---

## Notes

- This plan intentionally does not migrate M2 to RadioML or raw IQ. The project proposal asks for ELINT/PDW pulse descriptor processing, and the existing synthetic emitter library is the right local substrate for that requirement.
- The old `TemporalCNN` remains available because M4 robustness currently imports it directly.
- The strict anchor is the most important behavioral change: M2 should fail loudly if type, mode, threat, LPI, or latency regress.
