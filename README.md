# Open-Vocabulary Detection Study 🎯

Open-Vocabulary Object Detection (OVD) 연구를 위한 종합 가이드. 기존에 정해진 클래스 외에 새로운 클래스도 검출할 수 있는 기술을 다룹니다.

---

## 📚 목차

- [Open-Vocabulary Detection 개요](#1-open-vocabulary-detection-개요)
- [주요 아키텍처 비교](#2-주요-아키텍처-비교)
- [학습 전략](#3-학습-전략)
- [평가 지표](#4-평가-지표)
- [실전 활용 가이드](#5-실전-활용-가이드)
- [Zero-shot/Few-shot 학습](#6-zeroshot-fewshot-학습)
- [성능 벤치마크](#7-성능-벤치마크)

---

## 1. Open-Vocabulary Detection 개요

### 1.1 기본 개념

**Traditional Object Detection**:
- 고정된 클래스 세트 (예: COCO 80 classes)
- 새로운 클래스를 검출하려면 재학습 필요
- Supervised learning 기반

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

### 1.2 핵심 기술

1. **CLIP 기반 alignment**: 텍스트와 이미지 특징을 동일한 공간에서 학습
2. **Transformer architecture**: query-based detection
3. **Semantic-aware features**: class-agnostic detection + semantic matching

### 1.3 주요 용어

| 용어 | 설명 |
|------|--|
| **Zero-shot detection** | 학습되지 않은 클래스 검출 |
| **Few-shot detection** | 소수의 예시로 학습 후 검출 |
| **Open-set detection** | 미리 정의되지 않은 클래스 허용 |
| **Text-prompt** | "dog", "car", "electric bike" 등 |
| **Embedding space** | 텍스트와 이미지가 공유하는 특징 공간 |

---

## 2. 주요 아키텍처 비교

### 2.1 YOLO-World

**Overview**:
- **Speed**: Real-time (280 FPS)
- **Accuracy**: State-of-the-art
- **Training**: Two-stage (detection + contrastive)

**Architecture**:
```
1. Stage 1: Standard YOLOv8 training
2. Stage 2: CLIP-based text encoder integration
   - Text prompts → Text embeddings
   - Image features → Semantic alignment
   - Zero-shot inference
```

**Key Features**:
- YOLO 기반 fast inference
- CLIP 텍스트 인코더 통합
- Contrastive learning 기반 semantic alignment
- Two-stage training strategy

**Strengths**:
- ✅ 매우 빠른 inference (280 FPS)
- ✅ 높은 정확도
- ✅ 유연한 text prompt
- ✅ 실전 적용 가능

**Weaknesses**:
- ⚠️ CLIP 의존
- ⚠️ Two-stage training 복잡

---

### 2.2 Grounding DINO

**Overview**:
- **Speed**: Slower (8 FPS)
- **Accuracy**: SOTA
- **Training**: Pre-trained on large-scale data

**Architecture**:
```
Backbone (Swin Transformer)
  ↓
Encoder (Self-attention)
  ↓
Cross-attention (Image-Text)
  ↓
Detection Head
  ↓
Zero-shot Prediction
```

**Key Features**:
- Swin Transformer backbone
- Cross-attention for image-text matching
- Large-scale pre-training (LAION-400M)
- Dense prediction

**Strengths**:
- ✅ 매우 높은 정확도
- ✅ 강력한 zero-shot 성능
- ✅ 복잡한 prompt 처리 가능
- ✅ Multi-modal 이해도 높음

**Weaknesses**:
- ⚠️ 느린 inference 속도
- ⚠️ 큰 모델 크기
- ⚠️ 복잡한 training

---

### 2.3 CLIP-based Detection

**Overview**:
- **Speed**: Fast (~50-100 FPS)
- **Accuracy**: Good
- **Training**: Pre-trained CLIP 활용

**Architecture**:
```
CLIP Image Encoder → Image Features
CLIP Text Encoder → Text Embeddings
         ↓
    Similarity Matching
         ↓
  Region Proposal
         ↓
 Classification
```

**Key Features**:
- CLIP pre-trained model 직접 활용
- Region proposals + CLIP classification
- No fine-tuning 필요

**Strengths**:
- ✅ 즉시 사용 가능 (zero-shot)
- ✅ 간단한 구현
- ✅ 빠른 inference

**Weaknesses**:
- ⚠️ 정확도 제한적
- ⚠️ Region proposal 필요
- ⚠️ fine-tuning 시 성능 감소

---

### 2.4 Comparative Summary

| Model | Speed (FPS) | mAP (Zero-shot) | Training | Complexity | Best For |
|-------|-------|----|---------|-------------|------|---|
| **YOLO-World** | 280 | **38.5** | Two-stage | Medium | Real-time OVD |
| **Grounding DINO** | 8 | **42.0** | One-stage | High | Maximum accuracy |
| **CLIP-based** | 70 | 32.0 | None | Low | Quick deployment |

---

## 3. 학습 전략

### 3.1 YOLO-World Training (Two-stage)

#### Stage 1: Detection Foundation

```python
# Stage 1: Standard YOLOv8 training
model = YOLO('yolov8m.pt')

stage1_args = {
    'data': 'coco128.yaml',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    
    # Standard settings
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.05,
    
    # Disable contrastive initially
    'contrastive_learning': False,
    'open_vocabulary': False
}

model.train(**stage1_args)
```

**Goal**: Establish strong detection capabilities

#### Stage 2: Open-vocabulary Adaptation

```python
# Stage 2: Add contrastive learning
model = YOLO('yolov8m.pt')

stage2_args = {
    'data': 'open_vocabulary.yaml',
    'epochs': 50,
    'batch': 8,
    'imgsz': 640,
    
    # Contrastive learning enabled
    'contrastive_learning': True,
    'contrastive_lambda': 1.0,
    'open_vocabulary': True,
    
    # CLIP text encoder
    'text_encoder': 'clip_text',
    'text_prompts': ['dog', 'cat', 'bird', ...],  # Custom classes
    
    # Fine-tuning
    'freeze_text_encoder': False,
    'lr0': 0.0001,
    'freeze_backbone': True
}

model.train(**stage2_args)
```

**Goal**: Enable open-vocabulary detection

### 3.2 Grounding DINO Training

```python
from grounding_dino import GroundingDINO

model = GroundingDINO(
    backbone='swin_T_224',
    box_threshold=0.3,
    text_threshold=0.2
)

# Pre-training data
dataset = load_pretrain_data('laion400m')

# Training
model.train(
    data=dataset,
    epochs=10,
    batch_size=32,
    lr=1e-5,
    weight_decay=1e-4
)
```

**Key characteristics**:
- Large-scale pre-training
- Dense prediction
- Cross-attention architecture

### 3.3 Custom Text Prompts

**Basic prompts**:
```python
prompts = [
    "dog",
    "cat",
    "bicycle",
    "car"
]
```

**Complex prompts**:
```python
prompts = [
    "electric car",
    "small dog",
    "large bicycle",
    "standing person"
]
```

**Negative prompts**:
```python
prompts = [
    "dog",
    "not cat"  # Exclude cats
]
```

---

## 4. 평가 지표

### 4.1 주요 metric

| Metric | 설명 | 계산 방식 |
|--------|--|------|
| **mAP (Zero-shot)** | 평균 정밀도 (학습 안 한 클래스) | IoU threshold별 정밀도 평균 |
| **mAP (Seen)** | 기존 클래스 성능 | 일반적인 mAP 계산 |
| **mAP (Novel)** | 새로운 클래스 성능 | unseen classes 에 대한 mAP |
| **Inference Speed** | 초당 처리 프레임 | FPS 계산 |

### 4.2 Evaluation Protocol

**Zero-shot evaluation**:
```python
from grounding_dino import evaluate

# Test on novel categories
novel_classes = ['airplane', 'train', 'truck']  # Not in training

results = evaluate(
    model=model,
    images=test_images,
    prompts=novel_classes,
    iou_threshold=0.5
)

print(f"mAP (Zero-shot): {results['mAP']:.2f}")
print(f"mAP₅₀: {results['mAP50']:.2f}")
print(f"mAP₇₅: {results['mAP75']:.2f}")
```

### 4.3 Custom Dataset Evaluation

```python
class OVD evaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, image, text_prompt):
        """Evaluate detection results"""
        
        # Detect
        detections = self.model.predict(image, text_prompt)
        
        # Calculate precision/recall
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for detection in detections:
            if is_correct_match(detection, gt_boxes):
                true_positives += 1
            else:
                false_positives += 1
        
        false_negatives = len(gt_boxes) - true_positives
        
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detections': len(detections)
        }
```

---

## 5. 실전 활용 가이드

### 5.1 Zero-shot Inference

```python
from yolov8 import YOLO

class YOLOWorldInference:
    """YOLO-World inference with open-vocabulary support"""
    
    def __init__(self, model_path='yolo-world.pt'):
        self.model = YOLO(model_path)
        self.model.eval()
    
    def detect(self, image, text_prompt, conf_threshold=0.3):
        """
        Detect objects given a text prompt
        
        Args:
            image: Input image (numpy array)
            text_prompt: Text category description
            conf_threshold: Minimum confidence
        
        Returns:
            detections: List of detected objects
        """
        # Detect with text prompt
        with torch.no_grad():
            results = self.model.predict(
                source=image,
                classes=None,
                conf=conf_threshold,
                text_prompt=text_prompt
            )
        
        return results.detections
    
    def detect_multi_class(self, image, class_names, conf_threshold=0.3):
        """
        Detect multiple classes with custom names
        
        Args:
            image: Input image
            class_names: List of class names
            conf_threshold: Confidence threshold
        
        Returns:
            detections: List of detected objects
        """
        # Build prompt
        prompt = ', '.join(class_names)
        
        # Detect
        with torch.no_grad():
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                text_prompt=prompt
            )
        
        return results.detections
```

### 5.2 Custom Categories

```python
# Define custom categories
custom_classes = [
    'electric vehicle',
    'bicycle',
    'scooter',
    'motorcycle',
    'drone',
    'helicopter'
]

# Detect
detections = yolo_world.detect_multi_class(
    image=image,
    class_names=custom_classes,
    conf_threshold=0.4
)

# Result
for detection in detections:
    print(f"Detected: {detection['class']} at {detection['bbox']}")
```

### 5.3 Prompt Engineering

**Good prompts (specific)**:
```python
"detection of dogs"
"detection of cats"
"detection of electric cars"
```

**Better prompts (with attributes)**:
```python
"detection of large dogs"
"detection of small cats"
"detection of electric cars and bicycles"
```

**Avoid vague prompts**:
```python
"things"  # Too broad
"objects"  # Too general
```

### 5.4 Batch Processing

```python
def batch_inference(model, images, text_prompts):
    """Process multiple images with different prompts"""
    results = []
    
    for image, prompt in zip(images, text_prompts):
        detection = model.detect(image, prompt)
        results.append(detection)
    
    return results

# Or use parallel processing
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    results = list(executor.map(
        lambda args: model.detect(*args),
        zip(images, text_prompts)
    ))
```

---

## 6. Zero-shot/Few-shot 학습

### 6.1 Zero-shot Inference (No Training)

```python
# Use pre-trained YOLO-World
model = YOLO('yolo-world.pt')

# Direct inference on unseen classes
detections = model.detect(
    image=image,
    text_prompt="rare objects",
    conf=0.25
)
```

**Performance**:
- Good for common unseen classes
- May miss rare categories
- Depends on CLIP pre-training

### 6.2 Few-shot Learning

**Scenario**: 학습 데이터는 적지만 클래스는 많음

```python
def few_shot_adaptation(model, support_images, support_labels, target_prompt):
    """
    Few-shot adaptation for target class
    
    Args:
        model: Pre-trained OVD model
        support_images: 3-5 support images
        support_labels: Corresponding labels
        target_prompt: Target class name
    
    Returns:
        adapted_model: Few-shot adapted model
    """
    # Extract features from support set
    features = model.extract_features(support_images)
    
    # Compute support embeddings
    support_embeddings = compute_class_embedding(features, support_labels)
    
    # Update model with support information
    adapted_model = model.adapt(
        support_embeddings=support_embeddings,
        target_class=target_prompt,
        learning_rate=0.0001
    )
    
    return adapted_model
```

**Key components**:
- **Support set**: 3-5 examples per class
- **Feature extraction**: Use pre-trained backbone
- **Adaptation**: Fine-tune detection head only
- **Inference**: Fast adaptation to new classes

### 6.3 Class Prompt Learning

```python
class PromptLearning(nn.Module):
    """Learnable text prompts for specific classes"""
    
    def __init__(self, num_classes, prompt_length=16):
        super().__init__()
        self.num_classes = num_classes
        
        # Learnable prompt vectors
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_classes, prompt_length, 512)
        )
        
        # Projection to text embedding space
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
    
    def forward(self, class_indices):
        """Generate prompts for given classes"""
        prompts = self.prompt_embeddings[class_indices]
        embeddings = self.projection(prompts)
        
        return F.normalize(embeddings, p=2, dim=-1)
```

---

## 7. 성능 벤치마크

### 7.1 COCO Zero-shot Performance

| Task | Model | mAP | Speed (FPS) |
|------|-------|-----|-------------|
| **COCO (Zero-shot)** | YOLOv8 | 31.2 | 295 |
| **COCO (Zero-shot)** | Grounding DINO | 35.8 | 8 |
| **COCO (Zero-shot)** | YOLO-World | **38.5** | **280** |

### 7.2 Custom Categories Performance

| Category | YOLOv8 | Grounding DINO | YOLO-World |
|------|--------|------------|-- ------|
| **Electric vehicle** | ❌ N/A | ✅ 42.3 | ✅ **48.7** |
| **Drone** | ❌ N/A | ✅ 38.9 | ✅ **45.2** |
| **Fire hydrant** | ❌ N/A | ✅ 44.1 | ✅ **50.3** |
| **Speed limit sign** | ❌ N/A | ✅ 40.7 | ✅ **47.8** |

### 7.3 Speed Comparison

| Model | FPS | mAP | Latency (ms) |
|-------|-----|-----|-------------|
| **Grounding DINO** | 8 | 42.0 | 125 |
| **YOLO-World** | **280** | **38.5** | 3.6 |
| **YOLOv8** | 295 | 31.2 | 3.4 |

**Key insight**: YOLO-World is **35x faster** than Grounding DINO with comparable accuracy

---

## 🚀 Common Issues & Solutions

### Issue 1: Poor Zero-shot Performance

**Symptoms**:
- Low mAP on unseen categories
- Missed detections

**Solution**:
```python
# Improve text prompts
prompts = [
    "dog",
    "small dog",
    "large dog",
    "standing dog",
    "running dog"
]

# Use ensemble of prompts
detections = []
for prompt in prompts:
    det = model.detect(image, prompt)
    detections.append(det)

# Combine results
final = nms_combined(detections)
```

### Issue 2: Slow Inference

**Symptoms**:
- FPS much lower than expected
- Long inference time

**Solution**:
```python
# Optimize model
model.export(format='onnx')
model.export(format='engine')

# Reduce image size
results = model.predict(image, imgsz=416)

# Batch processing
results = model.predict([img1, img2, img3], batch_size=4)
```

### Issue 3: Hallucinations

**Symptoms**:
- Detects objects that don't exist
- False positives

**Solution**:
```python
# Increase confidence threshold
results = model.detect(image, prompt, conf=0.5)

# Use negative prompts
results = model.detect(
    image, 
    prompt="dog",
    negative_prompt="cat"
)

# Post-process results
filtered = remove_low_quality(results, quality_threshold=0.6)
```

---

## 📝 결론

### YOLO-World 장점

1. **Open-vocabulary**: Any text-based category
2. **Zero-shot**: No retraining needed
3. **Real-time**: YOLO speed (280 FPS)
4. **Flexible**: Custom prompts
5. **Efficient**: Pre-trained CLIP integration

### 단점

1. **Less mature**: YOLOv8 보다 연구 적음
2. **Prompt dependency**: Performance varies with prompts
3. **Training complexity**: Two-stage training
4. **Requires CLIP**: Additional dependencies

### 사용 추천

**추천 YOLO-World**:
- ✅ Open-vocabulary detection 필요
- ✅ Zero-shot inference 중요
- ✅ Custom categories 다수
- ✅ Real-time performance 필요

**추천 Grounding DINO**:
- ✅ Maximum accuracy 우선
- ✅ Complex prompts 필요
- ✅ Speed less critical
- ✅ Multi-modal understanding 필요

### Future Directions

**Next improvements**:
1. **Better prompts**: Adaptive prompt generation
2. **Few-shot learning**: Learn from examples
3. **Multimodal**: Add image captioning
4. **Distillation**: Smaller open-vocabulary models

---

*마지막 업데이트: 2026-03-30*
*참고: YOLO-World official, CLIP paper, Ultralytics documentation, Grounding DINO paper*
