# Open-Vocabulary Detection Study 🎯

Open-Vocabulary Object Detection (OVD) 연구를 위한 종합 가이드.

## 📚 목차

- [Open-Vocabulary Detection 개요](#1-open-vocabulary-detection-개요)
- [주요 아키텍처 비교](#2-주요-아키텍처-비교)
- [평가 지표](#4-평가-지표)
- [실전 활용 가이드](#5-실전-활용-가이드)
- [성능 벤치마크](#7-성능-벤치마크)

## 1. Open-Vocabulary Detection 개요

### 1.1 기본 개념

**Traditional Object Detection**:
- 고정된 클래스 세트 (예: COCO 80 classes)
- 새로운 클래스를 검출하려면 재학습 필요

**Open-Vocabulary Detection (OVD)**:
- **텍스트 기반 클래스 정의**: "개, 고양이, 차" 등 임의의 텍스트로 클래스 정의
- **Zero-shot inference**: 학습하지 않은 클래스도 검출 가능
- **Text-Image alignment**: CLIP 등의 pre-trained 모델 활용

```
Text Prompt → Text Encoder → Text Embeddings
                                                ↓
Image → Image Encoder → Feature Maps → Similarity → Detection
                                                ↓
                                        Open-set Objects
```

## 2. 주요 아키텍처 비교

### 2.1 YOLO-World vs Grounding DINO

| Model | Speed (FPS) | mAP (Zero-shot) | Training |
|-------|-------|--|----|
| **YOLO-World** | 280 | **38.5** | Two-stage |
| **Grounding DINO** | 8 | **42.0** | Pre-trained |

**YOLO-World 장점**:
- ✅ 매우 빠른 inference (280 FPS)
- ✅ 높은 정확도
- ✅ 실시간 처리 가능

**Grounding DINO 장점**:
- ✅ 최고 수준의 정확도
- ✅ 복잡한 prompt 처리 가능

## 3. 학습 전략

### 3.1 Two-stage Training

```python
# Stage 1: Standard YOLOv8 training
model = YOLO('yolov8m.pt')
model.train(data='coco128.yaml', epochs=100, batch=16)

# Stage 2: Add contrastive learning
model.train(
    contrastive_learning=True,
    open_vocabulary=True,
    text_encoder='clip_text',
    lr0=0.0001
)
```

## 4. 평가 지표

| Metric | 설명 |
|--------|--|
| **mAP (Zero-shot)** | 평균 정밀도 (학습 안 한 클래스) |
| **mAP (Seen)** | 기존 클래스 성능 |
| **Inference Speed** | 초당 처리 프레임 (FPS) |

**Evaluation Protocol**:
```python
# Test on novel categories
novel_classes = ['electric vehicle', 'drone', 'fire hydrant']

results = evaluate(
    model=model,
    images=test_images,
    prompts=novel_classes,
    iou_threshold=0.5
)

print(f"mAP (Zero-shot): {results['mAP']:.2f}")
print(f"Inference Speed: {results['fps']:.0f} FPS")
```

## 5. 실전 활용 가이드

### 5.1 Zero-shot Inference

```python
from yolov8 import YOLO

class YOLOWorldInference:
    def __init__(self, model_path='yolo-world.pt'):
        self.model = YOLO(model_path)
    
    def detect(self, image, text_prompt, conf=0.3):
        """Detect objects given a text prompt"""
        results = self.model.predict(
            source=image,
            conf=conf,
            text_prompt=text_prompt
        )
        return results.detections
    
    def detect_multi_class(self, image, class_names, conf=0.3):
        """Detect multiple classes"""
        prompt = ', '.join(class_names)
        results = self.model.predict(
            source=image,
            conf=conf,
            text_prompt=prompt
        )
        return results.detections
```

**Usage**:
```python
model = YOLOWorldInference()

# Single class
detections = model.detect(image, "dog")

# Multiple classes
detections = model.detect_multi_class(
    image, 
    ["dog", "cat", "bicycle"],
    conf=0.4
)

# Custom categories
detections = model.detect_multi_class(
    image,
    ["electric vehicle", "drone", "fire hydrant"]
)
```

## 6. Zero-shot/Few-shot 학습

### 6.1 Zero-shot Inference (No Training)

```python
# Use pre-trained YOLO-World
model = YOLO('yolo-world.pt')

# Direct inference on unseen classes
detections = model.detect(image, "rare objects", conf=0.25)
```

### 6.2 Prompt Engineering

**Good prompts**:
- "dog", "cat", "car"
- "electric car", "small dog"
- "large bicycle", "standing person"

**Avoid vague prompts**:
- "things" ❌
- "objects" ❌

## 7. 성능 벤치마크

### COCO Zero-shot Performance

| Task | Model | mAP | Speed (FPS) |
|------|-------|-----|--|
| **COCO (Zero-shot)** | YOLOv8 | 31.2 | 295 |
| **COCO (Zero-shot)** | Grounding DINO | 35.8 | 8 |
| **COCO (Zero-shot)** | YOLO-World | **38.5** | **280** |

### Custom Categories Performance

| Category | YOLOv8 | Grounding DINO | YOLO-World |
|------|--------|--|---|
| **Electric vehicle** | ❌ | 42.3 | **48.7** |
| **Drone** | ❌ | 38.9 | **45.2** |
| **Fire hydrant** | ❌ | 44.1 | **50.3** |

**Key insight**: YOLO-World is **35x faster** than Grounding DINO with comparable accuracy

---

## 📝 결론

### YOLO-World 장점

1. **Open-vocabulary**: Any text-based category
2. **Zero-shot**: No retraining needed
3. **Real-time**: YOLO speed (280 FPS)
4. **Flexible**: Custom prompts

### 사용 추천

**추천 YOLO-World**:
- ✅ Open-vocabulary detection 필요
- ✅ Zero-shot inference 중요
- ✅ Custom categories 다수
- ✅ Real-time performance 필요

---

*마지막 업데이트: 2026-03-30*
*참고: YOLO-World official, CLIP paper, Ultralytics documentation*
