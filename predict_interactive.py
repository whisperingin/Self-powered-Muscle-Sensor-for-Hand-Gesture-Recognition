import argparse
import functools
import json
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Title: A Real-Time Interactive Inference System for Surface EMG-Driven Hand Gesture Recognition

import numpy as np
import serial
import torch
import torch.nn as nn
import numpy as np

BIN_SYNC1 = 0xAA
BIN_SYNC2 = 0x55
BIN_STATUS_VALID = 0
BIN_STATUS_LOST = 1
BIN_PKT_SIZE = 17
LOST_SENTINEL = 65535

# ================================================================
# Model definition (consistent with the training script)
# ================================================================
class DynamicEMGNet(nn.Module):
    def __init__(self, n_channels, n_classes, params):
        super().__init__()
        n_layers  = params["n_layers"]
        filters   = params["filters"]
        kernel    = params["kernel"]
        fc_hidden = params["fc_hidden"]
        dropout1  = params["dropout1"]
        dropout2  = params["dropout2"]

        layers, in_ch = [], n_channels
        for _ in range(n_layers):
            out_ch = filters[len(layers) // 4]
            pad    = kernel // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch

        self.features   = nn.Sequential(*layers)
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


def expand_params_from_optuna(best_params):
    n_layers = int(best_params["n_layers"])
    f        = int(best_params["filter_base"])
    filters  = []
    for _ in range(n_layers):
        filters.append(f)
        f = min(f * 2, 256)
    return {
        "n_layers":     n_layers,
        "filters":      filters,
        "kernel":       int(best_params["kernel"]),
        "fc_hidden":    int(best_params["fc_hidden"]),
        "dropout1":     float(best_params["dropout1"]),
        "dropout2":     float(best_params["dropout2"]),
        "lr":           float(best_params["lr"]),
        "weight_decay": float(best_params["weight_decay"]),
        "batch_size":   int(best_params["batch_size"]),
    }


# ================================================================
# Utility functions (unchanged)
# ================================================================
def normalise_gesture_names(raw_map):
    out = {}
    if not isinstance(raw_map, dict):
        return out
    for k, v in raw_map.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out


def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Add a retry mechanism to handle transient file locks on Windows
    for _ in range(10):
        try:
            tmp.replace(path)
            break  # Exit the loop once replacement succeeds
        except PermissionError:
            time.sleep(0.02)  # If the file is busy, wait 20 ms and try again


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def read_one_bin_packet(ser):
    """
    Synchronize on and read one 17-byte binary packet from the serial port
    Returns:
        {
            "seq": int,
            "status": int,
            "values": np.array([ch1, ch2, ch3, ch4], dtype=np.float32)
        }
    Returns None on failure or timeout
    """
    while True:
        b = ser.read(1)
        if not b:
            return None

        if b[0] != BIN_SYNC1:
            continue

        b2 = ser.read(1)
        if not b2:
            return None

        if b2[0] != BIN_SYNC2:
            continue

        rest = ser.read(BIN_PKT_SIZE - 2)
        if len(rest) != BIN_PKT_SIZE - 2:
            return None

        pkt = b + b2 + rest

        crc_rx = int.from_bytes(pkt[15:17], byteorder="little")
        crc_calc = crc16_ccitt_false(pkt[:15])

        if crc_rx != crc_calc:
            # CRC mismatch; keep searching for the next packet
            continue

        seq = int.from_bytes(pkt[2:6], byteorder="little")
        status = pkt[6]
        ch1 = int.from_bytes(pkt[7:9], byteorder="little")
        ch2 = int.from_bytes(pkt[9:11], byteorder="little")
        ch3 = int.from_bytes(pkt[11:13], byteorder="little")
        ch4 = int.from_bytes(pkt[13:15], byteorder="little")

        values = np.asarray([ch1, ch2, ch3, ch4], dtype=np.float32)

        return {
            "seq": seq,
            "status": status,
            "values": values,
        }

def parse_csv4(line: str):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 4:
        return None
    try:
        return np.asarray([float(parts[i]) for i in range(4)], dtype=np.float32)
    except ValueError:
        return None


def build_raw_feature_window(raw_arr, channel_names):
    mapping = {f"CH{i+1}": raw_arr[:, i] for i in range(4)}
    # CH4_Band_Current is also mapped to the 4th column (the 4th serial column is CH4)
    mapping["CH4_Band_Current"] = raw_arr[:, 3]
    invalid = [n for n in channel_names if n not in mapping]
    if invalid:
        raise KeyError(f"Unsupported channel columns: {invalid}")
    return np.stack([mapping[n] for n in channel_names], axis=1).astype(np.float32)


# ================================================================
# Inference controller
# ================================================================
class InferenceController:
    def __init__(self, *, ser, model, channels, window_size, infer_stride,
                 z_mean, z_std_safe, gesture_names, device,
                 collect_seconds, subtract_mid, adc_mid,
                 confidence_threshold, state_path):
        self.ser                  = ser
        self.model                = model
        self.channels             = channels
        self.window_size          = window_size
        self.infer_stride         = infer_stride      # FIX-1 added
        self.z_mean               = z_mean
        self.z_std_safe           = z_std_safe
        self.gesture_names        = gesture_names
        self.device               = device
        self.collect_seconds      = collect_seconds
        self.subtract_mid         = subtract_mid
        self.adc_mid              = adc_mid
        self.confidence_threshold = confidence_threshold
        self.state_path           = state_path

        self._lock       = threading.Lock()
        self._active     = False
        self._stop_event = threading.Event()
        self._thread     = None

    def start(self):
        with self._lock:
            if self._active:
                return
            self._active = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        print("[CTRL] Inference loop started")

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self._active = False
        self._write({"phase": "idle", "timestamp": time.time()})
        print("[CTRL] Inference loop stopped")

    def _write(self, data: dict):
        atomic_write_json(self.state_path, data)

    def _collect(self):
        samples = []
        deadline = time.time() + self.collect_seconds
        next_tick = time.time()

        BIN_SYNC1 = 0xAA
        BIN_SYNC2 = 0x55
        BIN_PKT_SIZE = 17
        BIN_STATUS_VALID = 0

        def crc16_ccitt_false(data: bytes) -> int:
            crc = 0xFFFF
            for byte in data:
                crc ^= (byte << 8)
                for _ in range(8):
                    if crc & 0x8000:
                        crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                    else:
                        crc = (crc << 1) & 0xFFFF
            return crc

        def read_one_bin_packet():
            while True:
                if self._stop_event.is_set():
                    return None

                b1 = self.ser.read(1)
                if not b1:
                    return None

                if b1[0] != BIN_SYNC1:
                    continue

                b2 = self.ser.read(1)
                if not b2:
                    return None

                if b2[0] != BIN_SYNC2:
                    continue

                rest = self.ser.read(BIN_PKT_SIZE - 2)
                if len(rest) != (BIN_PKT_SIZE - 2):
                    return None

                pkt = b1 + b2 + rest

                crc_rx = int.from_bytes(pkt[15:17], byteorder="little")
                crc_calc = crc16_ccitt_false(pkt[:15])
                if crc_rx != crc_calc:
                    continue

                seq = int.from_bytes(pkt[2:6], byteorder="little")
                status = pkt[6]
                ch1 = int.from_bytes(pkt[7:9], byteorder="little")
                ch2 = int.from_bytes(pkt[9:11], byteorder="little")
                ch3 = int.from_bytes(pkt[11:13], byteorder="little")
                ch4 = int.from_bytes(pkt[13:15], byteorder="little")

                sample = np.asarray([ch1, ch2, ch3, ch4], dtype=np.float32)
                return seq, status, sample

        while time.time() < deadline:
            if self._stop_event.is_set():
                return None

            now = time.time()
            if now >= next_tick:
                self._write({
                    "phase": "collecting",
                    "countdown": round(max(0.0, deadline - now), 1),
                    "collect_seconds": self.collect_seconds,
                    "n_samples": len(samples),
                    "timestamp": now,
                })
                next_tick = now + 0.1

            pkt = read_one_bin_packet()
            if pkt is None:
                continue

            seq, status, sample = pkt

            # Enable during debugging if needed
            # print(f"[PKT] seq={seq} status={status} sample={sample}")

            if status != BIN_STATUS_VALID:
                # This is a padding packet for packet loss; skip it directly
                continue

            if self.subtract_mid:
                sample = sample - self.adc_mid

            samples.append(sample)

        return samples

    # ────────────────────────────────────────────────────────
    # FIX-1  Sliding-window averaged inference
    # Original: only take the last window_size samples and run inference once
    # New: split all data into multiple windows, run batch inference, and average probabilities
    # ────────────────────────────────────────────────────────
    def _infer(self, samples):
        if not samples:
            return None

        raw_arr = np.asarray(samples, dtype=np.float32)  # (N, 4)

        # Drop the first 1 second (estimated using the actual sampling rate, nominally 2000 Hz)
        WARMUP_SAMPLES = 2000  # 1 s × 2000 Hz
        if len(raw_arr) > WARMUP_SAMPLES:
            raw_arr = raw_arr[WARMUP_SAMPLES:]
        else:
            print(f"[WARN] Total sample count {len(raw_arr)} <= warmup {WARMUP_SAMPLES}; skipping this round")
            return None

        n = len(raw_arr)

        # Sliding-window split
        windows = []
        for start in range(0, n - self.window_size + 1, self.infer_stride):
            seg  = raw_arr[start : start + self.window_size]          # (W, 4)
            feat = build_raw_feature_window(seg, self.channels)       # (W, C)
            feat = (feat - self.z_mean[None, :]) / self.z_std_safe[None, :]
            windows.append(feat)

        # Not enough samples to form a full window
        if not windows:
            print(f"[WARN] Sample count {n} < window_size {self.window_size}; skipping this round")
            return None

        # Batch inference  (W, C, T)
        X = torch.from_numpy(
            np.stack(windows).transpose(0, 2, 1)
        ).float().to(self.device)

        with torch.no_grad():
            all_probs = torch.softmax(
                self.model(X), dim=1
            ).cpu().numpy()                      # (n_windows, n_classes)

        # Probability averaging (smoother than hard voting and gives a more accurate confidence estimate)
        mean_probs = all_probs.mean(axis=0)      # (n_classes,)
        pid  = int(np.argmax(mean_probs))
        conf = float(mean_probs[pid])

        return {
            "gesture_id": pid,
            "name":       self.gesture_names.get(pid, str(pid)),
            "confidence": conf,
            "all_probs":  {self.gesture_names.get(i, str(i)): float(p)
                           for i, p in enumerate(mean_probs)},
            "n_samples":  n,
            "n_windows":  len(windows),    # FIX-2 added for easier debugging
        }

    def _run(self):
        attempt = 0
        try:
            while not self._stop_event.is_set():
                attempt += 1

                samples = self._collect()
                if samples is None or self._stop_event.is_set():
                    break

                result = self._infer(samples)
                if result is None:
                    continue

                conf = result["confidence"]

                if conf >= self.confidence_threshold:
                    self._write({
                        "phase":                "result",
                        "gesture_id":           result["gesture_id"],
                        "name":                 result["name"],
                        "confidence":           conf,
                        "all_probs":            result["all_probs"],
                        "n_samples":            result["n_samples"],
                        "n_windows":            result["n_windows"],
                        "attempt":              attempt,
                        "confidence_threshold": self.confidence_threshold,
                        "timestamp":            time.time(),
                    })
                    print(
                        f"[OK  #{attempt:>3}] {result['name']:<12} "
                        f"conf={conf:.3f}  samples={result['n_samples']}  "
                        f"windows={result['n_windows']}"
                    )
                    attempt = 0
                    if self._stop_event.wait(timeout=0.8):
                        break
                else:
                    self._write({
                        "phase":                "retrying",
                        "gesture_id":           result["gesture_id"],
                        "name":                 result["name"],
                        "confidence":           conf,
                        "all_probs":            result["all_probs"],
                        "n_windows":            result["n_windows"],
                        "attempt":              attempt,
                        "confidence_threshold": self.confidence_threshold,
                        "timestamp":            time.time(),
                    })
                    print(
                        f"[RETRY #{attempt:>2}] {result['name']:<12} "
                        f"conf={conf:.3f} < {self.confidence_threshold:.2f}  "
                        f"windows={result['n_windows']}, retrying..."
                    )
                    if self._stop_event.wait(timeout=0.25):
                        break
        finally:
            with self._lock:
                self._active = False


# ================================================================
# HTTP handler (unchanged)
# ================================================================
def make_handler(controller: InferenceController, web_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path.startswith("/api/control"):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body   = json.loads(self.rfile.read(length))
                    cmd    = body.get("cmd", "")
                except Exception:
                    cmd = ""

                if cmd == "start":
                    controller.start()
                elif cmd == "stop":
                    controller.stop()

                resp = json.dumps({"ok": True, "cmd": cmd}).encode()
                self.send_response(200)
                self.send_header("Content-Type",   "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_error(404)

        def log_message(self, *_):
            pass

    return functools.partial(Handler, directory=str(web_dir))


# ================================================================
# Main entry point
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",                 default="model_cnn_tuned_2.pt")
    parser.add_argument("--port",                  required=True)
    parser.add_argument("--baud",                  type=int,   default=921600)
    parser.add_argument("--html",                  default="hand_model_2_en.html")
    parser.add_argument("--host",                  default="127.0.0.1")
    parser.add_argument("--http-port",             type=int,   default=8000)
    parser.add_argument("--no-browser",            action="store_true")
    parser.add_argument("--adc-mid",               type=float, default=2048.0)
    parser.add_argument("--collect-seconds",       type=float, default=4.0)
    parser.add_argument("--confidence-threshold",  type=float, default=0.80)
    parser.add_argument("--subtract-mid-from-raw", action="store_true")
    # FIX-1: add infer-stride argument; by default read the training stride from the model file
    parser.add_argument("--infer-stride",          type=int,   default=None,
                        help="Inference sliding-window stride; by default use the stride saved in the model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    channels = [str(x) for x in (ckpt.get("channels") or [])]
    if not channels:
        raise KeyError("The model file does not contain a channels field")

    n_channels  = int(ckpt.get("n_channels",  len(channels)))
    n_classes   = int(ckpt.get("n_classes",   7))
    window_size = int(ckpt.get("window_size", 300))

    # FIX-1: read stride from the model file; command line can override it
    infer_stride = args.infer_stride or int(ckpt.get("stride", window_size // 4))

    model_params = ckpt.get("model_params") or \
                   expand_params_from_optuna(ckpt["best_optuna_params"])

    gesture_names = normalise_gesture_names(
        ckpt.get("gesture_names",
                 {0:"Rest", 1:"Fist", 2:"Spread", 3:"Flexion",
                  4:"Extension", 5:"Pronation", 6:"Supination"})
    )

    z_mean     = np.asarray(ckpt.get("zscore_mean", np.zeros(n_channels)),
                             dtype=np.float32).reshape(-1)
    z_std      = np.asarray(ckpt.get("zscore_std",  np.ones(n_channels)),
                             dtype=np.float32).reshape(-1)
    z_std_safe = np.where(np.abs(z_std) < 1e-8, 1.0, z_std)

    if len(z_mean) != n_channels:
        raise ValueError("zscore_mean dimensions do not match the number of channels")

    model = DynamicEMGNet(n_channels=n_channels, n_classes=n_classes,
                          params=model_params).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Model loaded | channels: {channels} | "
          f"window_size={window_size} | infer_stride={infer_stride}")

    # Estimate the number of windows per voting round
    expected_samples = int(args.collect_seconds * 2000)
    expected_windows = max(0, (expected_samples - window_size) // infer_stride + 1)
    print(f"[INFO] Expected windows per round: ~{expected_windows} "
          f"({args.collect_seconds}s × 2000Hz = {expected_samples} samples)")

    html_path  = Path(args.html).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    web_dir    = html_path.parent
    state_path = web_dir / "gesture_state.json"
    atomic_write_json(state_path, {"phase": "idle", "timestamp": time.time()})

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    print(f"[INFO] Serial port: {args.port} @ {args.baud}")

    controller = InferenceController(
        ser=ser,
        model=model,
        channels=channels,
        window_size=window_size,
        infer_stride=infer_stride,         # FIX-1
        z_mean=z_mean,
        z_std_safe=z_std_safe,
        gesture_names=gesture_names,
        device=device,
        collect_seconds=args.collect_seconds,
        subtract_mid=args.subtract_mid_from_raw,
        adc_mid=args.adc_mid,
        confidence_threshold=args.confidence_threshold,
        state_path=state_path,
    )

    handler_class = make_handler(controller, web_dir)
    httpd         = ThreadingHTTPServer((args.host, args.http_port), handler_class)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    url = f"http://{args.host}:{args.http_port}/{html_path.name}"
    print(f"[INFO] HTTP service: {url}")
    print(f"[INFO] Confidence threshold: {args.confidence_threshold:.0%} | "
          f"collection duration: {args.collect_seconds}s")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        controller.stop()
        try:
            ser.close()
        except Exception:
            pass
        httpd.shutdown()


if __name__ == "__main__":
    main()



