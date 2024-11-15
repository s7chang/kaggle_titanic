import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 데이터 불러오기
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test2 = test.copy()

# 데이터 전처리 함수
def preprocess_data(df, is_train=True):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    df['T_partner'] = df["SibSp"] + df["Parch"]
    df['Alone'] = np.where(df['T_partner'] > 0, 0, 1)
    df['Words_Count'] = df['Name'].apply(lambda x: len(x.split()))
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
                                        'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    df['Cabin'] = df['Cabin'].fillna('U')
    df['Cabin'] = df['Cabin'].str[0]
    cabin_category = {'A': 9, 'B': 8, 'C': 7, 'D': 6, 'E': 5, 'F': 4, 'G': 3, 'T': 2, 'U': 1}
    df['Cabin'] = df['Cabin'].map(cabin_category)

    df['is_minor'] = np.where(df['Age'] <= 16, 1, 0)

    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
    df.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'T_partner'], axis=1, inplace=True)
    
    if is_train:
        df.drop("Survived", axis=1, inplace=True)

    return df

# 전처리 수행
y_train = train["Survived"]
train = preprocess_data(train, is_train=True)
test = preprocess_data(test, is_train=False)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train)
X_test_scaled = scaler.transform(test)

# PyTorch 텐서로 변환
X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 학습 및 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# PyTorch 모델 정의
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
num_epochs = 100
batch_size = 64
best_val_accuracy = 0

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    train_loss = 0.0
    correct_train = 0
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == batch_y).sum().item()
    
    train_loss /= X_train.size(0)
    train_accuracy = correct_train / X_train.size(0)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_predicted = (val_outputs > 0.5).float()
        val_accuracy = accuracy_score(y_val.numpy(), val_predicted.numpy())
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
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
    'PassengerId': test2['PassengerId'],
    'Survived': test_predicted.numpy()
})
submission.to_csv('submission.csv', index=False)
print('Prediction completed and saved to submission.csv')
