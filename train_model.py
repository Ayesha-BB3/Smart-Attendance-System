"""
Training Script - Siamese CNN with Triplet Loss
================================================
Trains face embedder from scratch on LFW dataset.

Usage:
    python train_model.py --lfw data/lfw --epochs 15 --steps 500
    python train_model.py --lfw data/lfw --quick    (for testing, ~10 min)
"""
import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf

from face_model import build_embedder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lfw', default='data/lfw', help='Path to LFW folder')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--steps', type=int, default=500, help='Steps per epoch')
    p.add_argument('--batch', type=int, default=32, help='Triplets per batch')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--margin', type=float, default=0.3)
    p.add_argument('--out', default='models/face_embedder.h5')
    p.add_argument('--quick', action='store_true', help='Use 100 people only (fast test)')
    return p.parse_args()


def load_lfw_into_memory(lfw_path, max_people=None, size=100):
    """Load all LFW images as uint8 arrays (fast in-memory training)."""
    lfw_path = Path(lfw_path)
    if not lfw_path.exists():
        print(f"[ERROR] LFW folder not found: {lfw_path}")
        print("Download: http://vis-www.cs.umass.edu/lfw/lfw.tgz")
        print("Extract to: data/lfw/  (so that data/lfw/Aaron_Peirsol/... exists)")
        sys.exit(1)
    
    people = {}
    person_dirs = sorted([d for d in lfw_path.iterdir() if d.is_dir()])
    
    for person_dir in person_dirs:
        jpgs = list(person_dir.glob('*.jpg'))
        if len(jpgs) < 2:
            continue
        
        loaded = []
        for p in jpgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            loaded.append(img)  # uint8
        
        if len(loaded) >= 2:
            people[person_dir.name] = loaded
            if max_people and len(people) >= max_people:
                break
    
    if not people:
        print("[ERROR] No identities with 2+ images found.")
        sys.exit(1)
    
    total = sum(len(v) for v in people.values())
    print(f"[INFO] Loaded {len(people)} identities | {total} images")
    mem_mb = total * size * size * 3 / 1024 / 1024
    print(f"[INFO] In-memory: ~{mem_mb:.0f} MB")
    return people


def augment(img_uint8):
    """Augment to simulate angle/glasses/beard/lighting variations."""
    img = img_uint8.copy()
    
    # Random rotation (±20°) - simulates head tilt/angle
    if random.random() > 0.3:
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((50, 50), angle, 1.0)
        img = cv2.warpAffine(img, M, (100, 100), borderMode=cv2.BORDER_REFLECT)
    
    # Random horizontal flip
    if random.random() > 0.5:
        img = np.flip(img, axis=1).copy()
    
    # Random brightness (glasses reflection, lighting)
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    # Random blur (glasses/soft focus)
    if random.random() > 0.8:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img.astype(np.float32) / 255.0


def make_batch(people, names, batch_size):
    anchors, positives, negatives = [], [], []
    while len(anchors) < batch_size:
        a_name = random.choice(names)
        imgs = people[a_name]
        i1, i2 = random.sample(range(len(imgs)), 2)
        n_name = random.choice([n for n in names if n != a_name])
        n_img = random.choice(people[n_name])
        
        anchors.append(augment(imgs[i1]))
        positives.append(augment(imgs[i2]))
        negatives.append(augment(n_img))
    
    return (
        np.array(anchors, dtype=np.float32),
        np.array(positives, dtype=np.float32),
        np.array(negatives, dtype=np.float32),
    )


def train_step(model, optimizer, anchor, positive, negative, margin):
    with tf.GradientTape() as tape:
        emb_a = model(anchor, training=True)
        emb_p = model(positive, training=True)
        emb_n = model(negative, training=True)
        
        pos_dist = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
        neg_dist = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
        loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss)


def main():
    args = parse_args()
    os.makedirs('models', exist_ok=True)
    
    print("=" * 60)
    print("Siamese CNN Training - From Scratch (no pre-trained)")
    print("=" * 60)
    print(f"TensorFlow : {tf.__version__}")
    print(f"LFW path   : {args.lfw}")
    print(f"Epochs     : {args.epochs}  |  Steps/epoch: {args.steps}")
    print(f"Batch      : {args.batch} triplets  |  LR: {args.lr}  |  Margin: {args.margin}")
    print("=" * 60)
    
    max_people = 100 if args.quick else None
    people = load_lfw_into_memory(args.lfw, max_people=max_people)
    names = list(people.keys())
    
    print("\n[INFO] Building model...")
    model = build_embedder()
    print(f"[INFO] Total parameters: {model.count_params():,}")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
    print("\n[INFO] Training started — close other apps to free CPU.\n")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        losses = []
        for step in range(1, args.steps + 1):
            a, p, n = make_batch(people, names, args.batch)
            loss = train_step(model, optimizer, a, p, n, args.margin)
            losses.append(loss)
            
            if step % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{args.steps} | "
                      f"Loss: {np.mean(losses[-50:]):.4f}")
        
        avg = np.mean(losses)
        print(f"\n>>> Epoch {epoch} done | Avg Loss: {avg:.4f}")
        
        if avg < best_loss:
            best_loss = avg
            model.save(args.out)
            print(f">>> Saved best model -> {args.out}\n")
        else:
            print()
    
    print("=" * 60)
    print(f"[DONE] Best loss: {best_loss:.4f}")
    print(f"[DONE] Model: {args.out}")
    print("=" * 60)
    print("\nNext: run 'python test_model.py' to verify the model.")


if __name__ == '__main__':
    main()