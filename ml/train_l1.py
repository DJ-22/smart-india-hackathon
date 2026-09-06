"""Level 1 -- fine-tune the wav2vec2 deepfake detector on IndicTTS + call channel.

  python ml/train_l1.py --run run1
  python ml/train_l1.py --run run2 --epochs 4 --lr 1.5e-5

Every run writes ml/checkpoints/l1_<run>/ plus a val-metrics json next to it, so
run2 can be compared against run1 on the identical val set before it is adopted.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import BASE_CKPT, CKPT_DIR, CROP_S
from ml.augment import opus_available
from ml.data import (Augmenter, ClipDataset, load_manifest, make_collator,
                     make_splits, split_report)
from ml.metrics import auc, eer


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="run1")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--base", default=BASE_CKPT)
    p.add_argument("--crop", type=float, default=CROP_S)
    p.add_argument("--max-minutes", type=float, default=45.0,
                   help="wall-clock guard; training stops cleanly past this")
    p.add_argument("--workers", type=int, default=6,
                   help="dataloader workers; 0 keeps everything in-process")
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main():
    a = build_args()
    import torch
    from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                              Trainer, TrainingArguments, TrainerCallback)

    print("cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    print("opus augmentation available:", opus_available())

    df = load_manifest()
    train_df, val_df, _ = make_splits(df)
    print(split_report(df))

    fe = AutoFeatureExtractor.from_pretrained(a.base)
    model = AutoModelForAudioClassification.from_pretrained(a.base, num_labels=2)
    for fn in ("freeze_feature_encoder", "freeze_feature_extractor"):
        if hasattr(model, fn):
            getattr(model, fn)()
            print("froze CNN via", fn)
            break

    # The upstream checkpoint ships id2label {0: 'fake', 1: 'real'} -- the exact
    # inverse of our manifest, where label 1 == TTS/fake. Rather than flip the
    # dataset (and hand every downstream consumer a trap), swap the two rows of
    # the pretrained classifier head so index 1 really means fake. The head stays
    # useful, and every other file in the repo can assume 1 == fake.
    from ml.infer import fake_index
    fake_i = fake_index(model.config.id2label)
    print("base id2label:", model.config.id2label, "-> fake index", fake_i)
    if fake_i != 1:
        with torch.no_grad():
            head = model.classifier
            perm = [1 - fake_i, fake_i]          # [real_row, fake_row]
            head.weight.copy_(head.weight[perm].clone())
            if head.bias is not None:
                head.bias.copy_(head.bias[perm].clone())
        print("swapped classifier head rows", perm)
    model.config.id2label = {0: "real", 1: "fake"}
    model.config.label2id = {"real": 0, "fake": 1}

    aug = None if a.no_augment else Augmenter(seed=0)
    ds_tr = ClipDataset(train_df, train=True, crop_s=a.crop, augment_fn=aug, seed=1)
    ds_va = ClipDataset(val_df, train=False, crop_s=a.crop, seed=2)
    print("train clips=%d  val clips=%d" % (len(ds_tr), len(ds_va)))

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        logits = np.asarray(logits, dtype=np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = (e / e.sum(axis=1, keepdims=True))[:, 1]
        y = np.asarray(p.label_ids)
        val_eer, _ = eer(y, prob)
        return {"eer": val_eer, "auc": auc(y, prob),
                "acc": float(((prob >= 0.5).astype(int) == y).mean())}

    out_dir = os.path.join(CKPT_DIR, "l1_" + a.run)
    # transformers 5 dropped warmup_ratio in favour of warmup_steps
    steps_per_epoch = max(1, len(ds_tr) // (a.bs * a.accum))
    warmup = max(20, int(0.1 * steps_per_epoch * a.epochs))
    kw = dict(
        output_dir=out_dir,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        warmup_steps=warmup,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eer",
        greater_is_better=False,
        logging_steps=25,
        disable_tqdm=True,                 # progress bars drown the log in a pipe
        dataloader_num_workers=a.workers,
        dataloader_persistent_workers=a.workers > 0,
        remove_unused_columns=False,
        report_to="none",
        seed=0,
    )
    try:
        targs = TrainingArguments(eval_strategy="epoch", **kw)
    except TypeError:
        targs = TrainingArguments(evaluation_strategy="epoch", **kw)

    class TimeGuard(TrainerCallback):
        """A mediocre checkpoint that exists beats a good one that doesn't."""

        def __init__(self, minutes):
            self.deadline = time.time() + minutes * 60

        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                print("\n[TimeGuard] wall-clock budget hit -- stopping cleanly",
                      flush=True)
                control.should_training_stop = True
                control.should_save = True
                control.should_evaluate = True
            return control

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=make_collator(fe), compute_metrics=compute_metrics,
        callbacks=[TimeGuard(a.max_minutes)],
    )

    t0 = time.time()
    trainer.train()
    mins = (time.time() - t0) / 60
    metrics = trainer.evaluate()
    print("train time: %.1f min" % mins)
    print("val:", json.dumps({k: round(float(v), 5) for k, v in metrics.items()
                              if isinstance(v, (int, float))}, indent=2))

    trainer.save_model(out_dir)
    fe.save_pretrained(out_dir)

    # load_best_model_at_end has already folded the best epoch into out_dir, so
    # the per-epoch checkpoints are now dead weight -- and they carry optimizer
    # state, which makes each run ~2.5 GB instead of ~360 MB. That matters here:
    # the repo lives in a synced OneDrive folder.
    import shutil
    for name in os.listdir(out_dir):
        if name.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(out_dir, name), ignore_errors=True)
    # training_args.bin is a pickle; inference never reads it, so don't ship it
    # alongside a model that gets copied between machines.
    ta = os.path.join(out_dir, "training_args.bin")
    if os.path.exists(ta):
        os.remove(ta)
    with open(os.path.join(out_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"run": a.run, "base": a.base, "epochs": a.epochs, "lr": a.lr,
                   "crop_s": a.crop, "augment": not a.no_augment,
                   "train_n": len(ds_tr), "val_n": len(ds_va),
                   "train_minutes": round(mins, 1),
                   "metrics": {k: float(v) for k, v in metrics.items()
                               if isinstance(v, (int, float))}}, f, indent=2)
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
