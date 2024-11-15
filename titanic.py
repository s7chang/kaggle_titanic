# %% [markdown]
# ## 캐글 타이타닉 : https://www.kaggle.com/competitions/titanic/overview
# ### 목표 : 전처리 방법 변경 및 모델을 Tensorflow 딥러닝 모델로 변경하여 제출 후 스코어 0.8 이상 도달하기

# %% [markdown]
# ### baseline

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
train_o = pd.read_csv('train.csv')
test_o = pd.read_csv('test.csv')

# %%
def preprocess_data(df):
    # 이름으로부터 타이틀 추출
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                    "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                    "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
    df['Title'] = df['Title'].map(title_mapping)

    # 나이 결측치 처리
    # fill missing age with median age for each title (Mr, Mrs, Miss, Others)
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))

    # Embarked 결측치 처리
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 범주형 변수 처리
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 26), 'Age'] = 1
    df.loc[(df['Age'] > 26) & (df['Age'] <= 36), 'Age'] = 2
    df.loc[(df['Age'] > 36) & (df['Age'] <= 62), 'Age'] = 3
    df.loc[df['Age'] > 62, 'Age'] = 4

    # 가족 수 계산
    df['Family'] = df['SibSp'] + df['Parch'] + 1

    # Pclass별 Cabin 최빈값으로 결측치 채우기
    df['Cabin'] = df.groupby('Pclass')['Cabin'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'U'))
    # Cabin 데이터 중 첫번째 알파벳만 처리
    df['Cabin'] = df['Cabin'].str[:1]

    # 원핫 인코딩을 사용
    df = pd.get_dummies(df, columns=['Cabin'], prefix='Cabin')

    # 원-핫 인코딩된 Cabin 열을 자동으로 포함하여 features 리스트 생성
    cabin_columns = [col for col in df.columns if col.startswith('Cabin_')]

    return df

# %%
# 훈련 데이터와 테스트 데이터 통합
train_o['is_train'] = 1  # 훈련 데이터 구분용
test_o['is_train'] = 0   # 테스트 데이터 구분용
test_o['Survived'] = -1  # Survived 열 추가, 이후 분리 시 제거

combined = pd.concat([train_o, test_o], ignore_index=True)

# 전처리 함수 적용
combined = preprocess_data(combined)

# 다시 훈련 데이터와 테스트 데이터로 분리
train = combined[combined['is_train'] == 1].drop(columns=['is_train'])
test = combined[combined['is_train'] == 0].drop(columns=['is_train', 'Survived'])
y = train['Survived']
train = train.drop(columns=['Survived'])

train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'SibSp', 'Parch'])
test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'SibSp', 'Parch'])

# 데이터 스케일링
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# PyTorch 텐서로 변환
X_tensor = torch.tensor(train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=None)

# %%
# 모델 정의
class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super(TitanicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# %%
# 모델 초기화
model = TitanicNet(input_size=train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 설정
num_epochs = 1000
batch_size = 256
best_val_accuracy = 0

train_accuracies = []  # 각 에폭의 train accuracy 값을 추가
val_accuracies = []    # 각 에폭의 val accuracy 값을 추가

for epoch in range(num_epochs):
    # 학습 모드 설정
    model.train()
    permutation = torch.randperm(X_train.size()[0])

    train_loss = 0.0
    correct_train = 0
    
    # 미니배치 학습
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        # Train loss와 accuracy 계산
        train_loss += loss.item() * batch_x.size(0)
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == batch_y).sum().item()
    
    # 에폭별 평균 train loss와 accuracy 계산
    train_loss /= X_train.size(0)
    train_accuracy = correct_train / X_train.size(0)

    # 검증 모드 설정
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_predicted = (val_outputs > 0.5).float()
        val_accuracy = accuracy_score(y_val, val_predicted)
    
    # 에폭별 결과 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # 최적의 모델 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pt")

    # 학습 및 검증 정확도 계산 코드 ...
    train_accuracies.append(train_accuracy)  # train_accuracy는 각 에폭에서 계산된 값
    val_accuracies.append(val_accuracy)      # val_accuracy는 각 에폭에서 계산된 값


print(f"Best model saved with val_accuracy: {best_val_accuracy:.4f}")

# %%
# 정확도 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_accuracies, label="Train Accuracy", linestyle="-", marker="o", markersize=3)
plt.plot(range(num_epochs), val_accuracies, label="Validation Accuracy", linestyle="-", marker="x", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# %%
# 테스트 데이터 예측
best_model = TitanicNet(input_size=train.shape[1])
best_model.load_state_dict(torch.load("best_model.pt"))
best_model.eval()

with torch.no_grad():
    test_outputs = best_model(X_test_tensor)
    test_predicted = (test_outputs > 0.5).float().squeeze()

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': test_o['PassengerId'],
    'Survived': test_predicted.int().numpy()
})
submission.to_csv('submission.csv', index=False)
print('Prediction completed and saved to submission.csv')


