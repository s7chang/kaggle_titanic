# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 데이터 전처리 함수 정의
def preprocess_data(df):
    # Title 추출
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, 
                     "Mlle": 3, "Countess": 3, "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, 
                     "Mme": 3, "Capt": 3, "Sir": 3 }
    df['Title'] = df['Title'].map(title_mapping)

    # Age 결측치 처리
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))

    # Embarked 결측치 처리
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 성별 변환
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # 가족 수 계산
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Fare 결측치 처리
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 불필요한 열 제거
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

# 데이터 전처리
train = preprocess_data(train)
test = preprocess_data(test)

# 특성(X)과 레이블(y) 분리
X = train.drop('Survived', axis=1)
y = train['Survived']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# PyTorch 텐서로 변환
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

# 훈련 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

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

# 모델 초기화
input_size = X_train.shape[1]
model = TitanicNet(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 500
batch_size = 64
best_val_accuracy = 0

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

print(f"Best model saved with val_accuracy: {best_val_accuracy:.4f}")

# 테스트 데이터 예측
best_model = TitanicNet(input_size)
best_model.load_state_dict(torch.load("best_model.pt"))
best_model.eval()

with torch.no_grad():
    test_outputs = best_model(X_test_tensor)
    test_predicted = (test_outputs > 0.5).int().squeeze()

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Survived': test_predicted.numpy()
})
submission.to_csv('submission.csv', index=False)
print('Prediction completed and saved to submission.csv')
