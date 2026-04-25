"""Background HF checkpoint uploader — TrainerCallback.

On every `on_save` (i.e. HF Trainer writes a new checkpoint-<N>/ dir),
fire off a background thread that pushes that dir to an HF model repo.
Training is never blocked: the thread uploads concurrently with the next
training steps.

Design choices:
- Uses `HfApi.upload_folder` with `run_as_future=True` under the hood via
  `threading.Thread` (keeps the file handle local, no stdio churn).
- One thread per save; orphaned or slow uploads don't block new saves.
- Target repo: Hula0401/cadrille-<run-tag>. Created (private) if missing.
- Each checkpoint lands at repo subdir `checkpoint-<N>/`. HF's folder upload
  is additive so re-runs of the same step (e.g. crash + resume) overwrite.

Failure modes:
- No HF_TOKEN with write scope → log error, skip upload, training continues.
- Network hiccup mid-upload → HF client retries internally; if it still
  fails, we log and move on. Training isn't affected.
"""
from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path

from transformers import TrainerCallback


class HFCheckpointUploadCallback(TrainerCallback):
    def __init__(self, repo_id: str, token: str | None = None,
                 private: bool = True, path_in_repo_prefix: str = ''):
        self.repo_id = repo_id
        self.token = token or os.environ.get('HF_TOKEN')
        self.private = private
        self.path_in_repo_prefix = path_in_repo_prefix.rstrip('/')
        self._threads: list[threading.Thread] = []
        self._repo_verified = False

    def _ensure_repo(self):
        if self._repo_verified:
            return
        if not self.token:
            print('[hf-upload] HF_TOKEN missing; uploads disabled', flush=True)
            return
        try:
            from huggingface_hub import HfApi
            HfApi().create_repo(self.repo_id, repo_type='model',
                                private=self.private, exist_ok=True,
                                token=self.token)
            self._repo_verified = True
            print(f'[hf-upload] target repo ready: {self.repo_id} '
                  f'(private={self.private})', flush=True)
        except Exception as e:
            print(f'[hf-upload] repo create failed: {e}', flush=True)

    def _upload_dir(self, local_dir: str, step: int):
        """Runs in a background thread."""
        try:
            from huggingface_hub import HfApi
            prefix = f'{self.path_in_repo_prefix}/' if self.path_in_repo_prefix else ''
            path_in_repo = f'{prefix}checkpoint-{step}'
            HfApi().upload_folder(
                folder_path=local_dir,
                repo_id=self.repo_id,
                repo_type='model',
                path_in_repo=path_in_repo,
                token=self.token,
                commit_message=f'ckpt-{step} auto-upload from SFT',
            )
            print(f'[hf-upload] pushed {local_dir} -> {self.repo_id}/{path_in_repo}',
                  flush=True)
        except Exception as e:
            print(f'[hf-upload] upload failed for step {step}: {e}', flush=True)
            traceback.print_exc()

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if not self.token:
            return

        self._ensure_repo()
        if not self._repo_verified:
            return

        # HF Trainer saves to args.output_dir/checkpoint-<global_step>
        step = state.global_step
        ckpt_dir = Path(args.output_dir) / f'checkpoint-{step}'
        if not ckpt_dir.is_dir():
            print(f'[hf-upload] no dir at {ckpt_dir}; skipping', flush=True)
            return

        t = threading.Thread(
            target=self._upload_dir,
            args=(str(ckpt_dir), step),
            daemon=True,
            name=f'hf-upload-{step}',
        )
        t.start()
        self._threads.append(t)
        print(f'[hf-upload] started background upload of checkpoint-{step}', flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._threads:
            return
        print(f'[hf-upload] waiting for {sum(1 for t in self._threads if t.is_alive())} '
              f'pending upload(s) to drain ...', flush=True)
        for t in self._threads:
            t.join(timeout=600)  # cap each at 10 min
