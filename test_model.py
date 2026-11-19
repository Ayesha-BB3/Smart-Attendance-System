"""
Test Model - Verify trained model quality
==========================================
Run AFTER training to check if model separates identities well.
"""
import sys
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path


def preprocess(path, size=100):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return (img.astype(np.float32) / 255.0)[None, ...]


def distance(e1, e2):
    return float(np.linalg.norm(e1 - e2))


def main():
    lfw = Path('data/lfw')
    model_path = 'models/face_embedder.h5'
    
    if not lfw.exists():
        print(f"[ERROR] LFW not found at {lfw}")
        return
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}. Train first!")
        return
    
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.\n")
    
    # Collect 5 same-person pairs
    same_pairs = []
    for p_dir in lfw.iterdir():
        if not p_dir.is_dir():
            continue
        imgs = list(p_dir.glob('*.jpg'))
        if len(imgs) >= 2:
            same_pairs.append((imgs[0], imgs[1], p_dir.name))
            if len(same_pairs) >= 5:
                break
    
    print("=" * 60)
    print("Test 1: SAME person (distance should be LOW)")
    print("=" * 60)
    same_dists = []
    for p1, p2, name in same_pairs:
        e1 = model.predict(preprocess(p1), verbose=0)[0]
        e2 = model.predict(preprocess(p2), verbose=0)[0]
        d = distance(e1, e2)
        same_dists.append(d)
        print(f"  {name[:25]:25s}  dist = {d:.4f}")
    mean_same = np.mean(same_dists)
    print(f"\n  Average same-person distance: {mean_same:.4f}\n")
    
    # 5 different-person pairs
    all_names = [p.name for p in lfw.iterdir()
                 if p.is_dir() and len(list(p.glob('*.jpg'))) >= 1]
    print("=" * 60)
    print("Test 2: DIFFERENT people (distance should be HIGH)")
    print("=" * 60)
    diff_dists = []
    for _ in range(5):
        n1, n2 = np.random.choice(all_names, 2, replace=False)
        p1 = next((lfw / n1).glob('*.jpg'))
        p2 = next((lfw / n2).glob('*.jpg'))
        e1 = model.predict(preprocess(p1), verbose=0)[0]
        e2 = model.predict(preprocess(p2), verbose=0)[0]
        d = distance(e1, e2)
        diff_dists.append(d)
        print(f"  {n1[:15]} vs {n2[:15]:15s}  dist = {d:.4f}")
    mean_diff = np.mean(diff_dists)
    print(f"\n  Average different-person distance: {mean_diff:.4f}\n")
    
    # Verdict
    sep = mean_diff - mean_same
    recommended_threshold = (mean_same + mean_diff) / 2
    
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Mean same-person distance:      {mean_same:.4f}")
    print(f"  Mean different-person distance: {mean_diff:.4f}")
    print(f"  Separation:                     {sep:.4f}")
    print(f"  Recommended threshold:          {recommended_threshold:.4f}")
    print()
    
    if sep > 0.5:
        print("  [EXCELLENT] Strong separation. Ready for integration.")
    elif sep > 0.3:
        print("  [GOOD] Model trained well. You can proceed.")
    elif sep > 0.15:
        print("  [OK] Needs more training. Try more epochs.")
    else:
        print("  [POOR] Model not trained enough. Train more.")
    
    print("\n" + "=" * 60)
    print("Note down the 'Recommended threshold' — Phase 4 me use hoga.")
    print("=" * 60)


if __name__ == '__main__':
    main()