# Task2 - VCL팀

2024.09.12 작성

베이스라인과 다르게 적용한 코드를 요약

1. dataset.py
2. train.py
3. networks.py
4. train.sh

---

### 1. dataset.py

- 학습 데이터 증강 및 전처리
	```python
	self.transform_image = transforms.Compose([
				transforms.CenterCrop(200),
				transforms.Resize((224,224)),
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(),
				transforms.ColorJitter(contrast=0.2),
				transforms.RandomAffine(degrees=30, scale=(0.6, 1.4)),
				transforms.ToTensor(),
			])
	```
- Validation 데이터 전처리
	```python
	self.transform_image = transforms.Compose([
                transforms.CenterCrop(200),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
	```

- 공통
	서로 다른 5가지 색공간 표현을 모델의 입력에 사용
	- **HSV, LAB, HLS, YUV, RGB**
		1. HSV : Hue(색조) - Saturation(채도) - Brightness(밝기)
		2. LAB : Lightness(명도) - A(녹색에서 적색의 보색) - B(황색에서 청색의 보색)
		3. HLS : Hue(색조) - Lightness(명도) - Saturation(채도)
		4. YUV : Luminance(휘도) - U,V(색차:Chrominance)
		5. RGB : Red(빨간색) - Green(초록색) - Blue(파란색)

---

### 2. train.py

1. ImbalancedDatasetSampler 사용하여 데이터 불균형 문제 해결
2. SAM, AdamP 옵티마이져 사용
3. CosineAnnealingLR 스케줄러 사용
4. CrossEntropyLoss(label smoothing=0.1) 사용
---

### 3. networks.py

1. efficientformerv2-s1 모델을 백본으로 사용(6.1M params)
2. 레이어 추가
	- MLP1
		- Linear(224, 112)
		- BatchNorm2d(112)
		- LeakyRelu(0.2)
		- Dropout(0.5)   
	- MLP2
		- Linear(112, 56)
		- BatchNorm2d(56)
		- LeakyRelu(0.2)
		- Dropout(0.3)   
	- classifier
		- Linear(56, 18)
---

### 4. train.sh

1. Learning Rate = 2e-3
2. Batch Size = 64