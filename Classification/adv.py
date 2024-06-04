import torch
import copy
import os
from collections import OrderedDict

import arg_parser
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
import torch.nn.functional as F

def generate_adversarial_examples(forget_loader, model, epsilon, alpha, num_steps, num_adv_examples):
    model.eval()  # 평가 모드로 전환
    adversarial_examples = []

    # 1. adversarial_examples 리스트 초기화 (D̄f ← ∅)
    for i, (data, target) in enumerate(forget_loader):
        data, target = data.to(device), target.to(device)
        original_data = data.clone().detach()

        # 2. 데이터 배치의 각 샘플에 대해 (for i in range Nf do)
        for j in range(num_adv_examples):
            data = original_data.clone().detach().requires_grad_(True)
            
            # 4. 무작위로 다른 타겟 라벨 샘플링 (Randomly sample ȳ(i) ≠ y(i))
            random_target = torch.randint(0, num_classes, target.shape).to(device)
            while torch.any(random_target == target):
                random_target = torch.randint(0, num_classes, target.shape).to(device)

            # 5. num_adv_examples 수만큼 적대적 예제 생성 (for j in range Nadv do)
            for step in range(num_steps):
                output = model(data)
                
                # 6. 손실 계산 및 역전파 (x'(j)f ← L2-PGD(x(i), ȳ(i)) (Eq. 1))
                loss = F.cross_entropy(output, random_target)
                model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                
                # 적대적 예제 업데이트
                data = data + alpha * data_grad.sign()
                data = torch.clamp(data, original_data - epsilon, original_data + epsilon)
                data = torch.clamp(data, 0, 1)  # 이미지 데이터의 경우 픽셀 값은 [0, 1]로 한정

            # 7. 생성된 적대적 예제를 리스트에 추가 (D̄f ← D̄f ∪ {(x'(j)f , ȳ(j))})
            adversarial_examples.append((data.detach(), random_target.detach()))

    # 10. 최종적으로 적대적 예제 리스트 반환 (return D̄f)
    print(f"Total number of adversarial examples: {len(adversarial_examples)}")
    return adversarial_examples

# 파라미터 설정
epsilon = 0.3  # 공격 강도
alpha = 2/255  # 스텝 크기
num_steps = 40  # 스텝 수
num_adv_examples = 20  # 각 데이터 포인트당 생성할 적대적 예제 수
num_classes = 10  # 클래스 수 (예: CIFAR-10의 경우 10)
