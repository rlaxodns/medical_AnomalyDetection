import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from tqdm import tqdm

# 한글 폰트 설정
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ==================================================================
#                       PART 1: 이상 탐지 모델 (AE)
# ==================================================================

def pad_sequences_numpy(sequences, maxlen, padding='post', truncating='post', dtype='float32'):
    num_samples = len(sequences)
    num_features = sequences[0].shape[1] if num_samples > 0 else 0
    padded_array = np.zeros((num_samples, maxlen, num_features), dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) == 0: continue
        if len(seq) > maxlen: seq = seq[-maxlen:] if truncating == 'pre' else seq[:maxlen]
        if len(seq) < maxlen:
            if padding == 'pre': padded_array[i, -len(seq):, :] = seq
            else: padded_array[i, :len(seq), :] = seq
        else: padded_array[i, :, :] = seq
    return padded_array

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(latent_dim, input_dim, num_layers, batch_first=True)
    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        encoder_output = hidden.permute(1, 0, 2)[:, -1, :]
        decoder_input = encoder_output.unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed_x, _ = self.decoder_lstm(decoder_input)
        return reconstructed_x

# --- 1-1. 데이터 로드 및 전처리 ---
print("--- 1-1. 전체 데이터 로드 및 전처리 시작 ---")
TOTAL_DATA_PATH = r'C:\Users\240913\Desktop\hero\toto.csv'
CHUNKSIZE = 100000
aaa=pd.read_csv(TOTAL_DATA_PATH, index_col=False)
print(aaa.shape)
# 모델 학습에 사용할 피처 목록 정의
features_to_keep = ['일련_번호', '진료_입원_일자', '진료과_코드_x', '순번_번호', '환자_구분', '환자_구분_명칭', '진료_나이',
       '성별', '진단_질병_코드', '진단_질병_명', '주진단_여부', '환자구분', '진료과_코드_y',
       '혈압(수축기)', '혈압(이완기)', '맥박수', '호흡수', '체온', '신장', '음주_유무', '혈액형',
       '혈액형_RH', '호흡기장애_무', '호흡기장애_유', '호흡기장애(호흡곤란)유무', '호흡기장애(가래)유무',
       '호흡기장애(기침)유무', '호흡기장애(이상호흡음)유무', '호흡기장애(폐잡음)유무', '호흡기장애(객혈)유무',
       '호흡기장애(청색증)유무', '호흡기장애(기관절개관)유무', '호흡기장애(기타)유무', '센터과_코드', '순환기장애_무',
       '순환기장애_유', '순환기장애(심계항진)유무', '순환기장애(흉통)유무', '순환기장애(청색증)유무',
       '순환기장애(호흡곤란)유무', '순환기장애(식은땀)유무', '순환기장애(부정맥)유무', '순환기장애(심잡음)유무',
       '순환기장애(기타)유무', '음주(병/회)', '음주(회/월)', '음주_기간_년수', '음주_종류', '소화기장애_무',
       '소화기장애_유', '소화기장애(연하곤란)유무', '소화기장애(오심)유무', '소화기장애(구토)유무', '소화기장애(토혈)유무',
       '소화기장애(소화장애)유무', '소화기장애(복부팽만)유무', '소화기장애(복부동통)유무', '소화기장애(점액변)유무',
       '소화기장애(인공장루)유무', '소화기장애(혈변)유무', '소화기장애(기타)유무', '청력장애_무', '청력장애_유',
       '청력장애(청력저하)유무', '청력장애(보청기)유무', '청력장애(이명)유무', '청력장애(청각상실(Rt))유무',
       '청력장애(청각상실(Lt))유무', '청력장애(기타)유무', '정서상태_안정', '정서상태_불안', '정서상태_슬픔',
       '정서상태_분노', '정서상태_우울', '정서상태_흥분', '정서상태_안절부절', '정서상태_기타', '시력장애_무',
       '시력장애_유', '시력장애(안경)유무', '시력장애(렌즈)유무', '시력장애(의안(Rt))유무',
       '시력장애(의안(Lt))유무', '시력장애(기타)유무', '신경계장애(신경근육)이상없음유무', '신경계장애(신경근육)무감각유무',
       '신경계장애(신경근육)저린감유무', '신경계장애(신경근육)동통유무', '신경계장애(신경근육)저하유무',
       '신경계장애(신경근육)부위유무', '수술_경험_유무', '통증_무', '통증_유', '통증부위(두통)유무',
       '통증부위(흉통)유무', '통증부위(복통)유무', '통증부위(요통)유무', '통증부위(사지통)유무', '통증부위(기타)유무',
       '통증정도(둔함)유무', '통증정도(쑤심)유무', '통증정도(퍼짐)유무', '통증정도(예리함)유무',
       '통증정도(찌르는듯함)유무', '통증정도(쥐어짜는듯함)유무', '신경계장애(마비)_무', '신경계장애(마비)_유',
       '신경계장애(마비)오른팔유무', '신경계장애(마비)오른발유무', '신경계장애(마비)왼팔유무', '신경계장애(마비)왼발유무',
       '신경계장애(마비)(기타)유무', '과거병력유무', '고혈압', '당뇨', '만성폐질환(결핵)', '만성간담췌질환(간염)',
       '심장질환', '뇌졸중', '암', '기타', '고지혈증', '코골이', '만성폐질환(이외)', '만성간담췌질환(이외)',
       '혈액암(발병기간상관없음)', '암(5년이내)', '암(5년이후)', '위장관질환', '만성신장질환', '갑상선질환', '이식',
       '섬망(과거입원시)', '과거_입원', '수면장애_무', '수면장애_유', '수면제복용_무', '수면제복용_유',
       '흡연(갑/일)', '흡연_유무', '흡연_기간_년수', '체중']
processed_chunks = []

try:
    chunk_iterator = pd.read_csv(TOTAL_DATA_PATH, chunksize=CHUNKSIZE, encoding='utf-8-sig', low_memory=False)

    for chunk in tqdm(chunk_iterator, desc="청크 처리 중"):
        chunk_subset = chunk[features_to_keep]
        processed_chunks.append(chunk_subset)
        
except FileNotFoundError:
    print(f"오류: '{TOTAL_DATA_PATH}' 경로에 파일이 없습니다. 경로를 다시 확인해주세요.")
    exit()


print("\n처리된 모든 청크를 하나로 결합합니다...")
total_df = pd.concat(processed_chunks, ignore_index=True)
total_df.sort_values(by=['일련_번호', '진료_입원_일자', '순번_번호'], inplace=True)


# 전처리: 범주형 데이터를 먼저 숫자로 변환
print("\n--- 범주형 데이터 인코딩 (숫자 변환) ---")
categorical_cols = total_df.select_dtypes(include=['object', 'category']).columns

for col in tqdm(categorical_cols, desc="범주형 피처 인코딩 중"):
    if col != '진단_질병_코드':
         total_df[col] = LabelEncoder().fit_transform(total_df[col].astype(str))

numeric_cols = [col for col in total_df.select_dtypes(include=np.number).columns if col not in ['일련_번호', '순번_번호']]
total_df[numeric_cols] = total_df[numeric_cols].fillna(0)
scaler = MinMaxScaler()
total_df[numeric_cols] = scaler.fit_transform(total_df[numeric_cols])

# --- 1-2. 오토인코더 학습 데이터(정상군) 준비 ---
print("\n--- 1-2. 오토인코더 학습 데이터 준비 ---")
parkinson_code = ['G20', 'G20.03', 'G20.04', 'G218', 'G20.05', 'G211',
                  'G214', 'G22', 'G20.01', 'G219', 'G232', 'I456.06',
                  'G212', 'F023']
stroke_codes = ['I64',  'I694',  'G463',  'G464',  'Z823']
all_target_codes = parkinson_code + stroke_codes

normal_patient_ids = total_df[~total_df['진단_질병_코드'].isin(all_target_codes)]['일련_번호'].unique()
normal_df = total_df[total_df['일련_번호'].isin(normal_patient_ids)]


features_for_lstm = [
    # 공통/신경계
    '신경계장애(마비)_유', '신경계장애(마비)오른팔유무', '신경계장애(마비)오른발유무',
    '신경계장애(마비)왼팔유무', '신경계장애(마비)왼발유무', '신경계장애(마비)(기타)유무',
    '신경계장애(신경근육)무감각유무', '신경계장애(신경근육)저린감유무',
    '신경계장애(신경근육)동통유무', '신경계장애(신경근육)저하유무',
    # 뇌졸중 관련
    '뇌졸중', '고혈압', '당뇨', '심장질환', '고지혈증', '만성신장질환',
    '혈압(수축기)', '혈압(이완기)', '맥박수', '체중', '신장',
    '흡연(갑/일)', '흡연_유무', '흡연_기간_년수',
    '음주_유무', '음주(병/회)', '음주(회/월)', '음주_기간_년수',
    '통증부위(두통)유무', '통증부위(사지통)유무',
    # 파킨슨 관련
    '수면장애_유', '수면제복용_유',
    '정서상태_우울', '정서상태_불안', '정서상태_안절부절',
    '청력장애_유', '청력장애(청력저하)유무', '청력장애(이명)유무',
    '시력장애_유', '시력장애(기타)유무',
    # 기타 공통 위험 요인
    '과거병력유무', '암', '주진단_여부'
]

grouped_normal = normal_df.groupby('일련_번호')
sequences_normal = [group[features_for_lstm].values for _, group in tqdm(grouped_normal, desc="정상군 시퀀스 생성 중")]
MAX_SEQ_LENGTH = int(np.percentile([len(s) for s in sequences_normal], 95)) if sequences_normal else 50
X_train_AE = pad_sequences_numpy(sequences_normal, maxlen=MAX_SEQ_LENGTH)
print(f"AE 학습 데이터 shape: {X_train_AE.shape}")

# --- 1-3. 오토인코더(AE) 모델 학습 ---
print("\n--- 1-3. 오토인코더(AE) 모델 학습 시작 ---")
device = torch.device('cpu')
INPUT_DIM = X_train_AE.shape[2]
model_AE = LSTMAutoencoder(INPUT_DIM, latent_dim=16, num_layers=2).to(device)
criterion = nn.MSELoss(reduction='none') # 개별 오차 계산을 위해 reduction='none'
optimizer = optim.Adam(model_AE.parameters(), lr=0.001)

train_tensor = torch.from_numpy(X_train_AE).float()
train_dataset = TensorDataset(train_tensor, train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(10): # Epochs는 10으로 줄여서 시연
    for inputs, targets in train_loader:
        outputs = model_AE(inputs)
        loss = criterion(outputs, targets).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"Epoch [{epoch+1}/10], AE Loss: {loss.item():.6f}")
print("오토인코더 학습 완료!")

# --- 1-4. AE를 이용한 '오진 의심 후보' 목록 생성 ---
print("\n--- 1-4. AE를 이용한 '오진 의심 후보' 목록 생성 시작 ---")
# 탐지 대상: 파킨슨 또는 뇌졸중 환자 전체
target_population_df = total_df[total_df['진단_질병_코드'].isin(all_target_codes)]
target_patient_ids = target_population_df['일련_번호'].unique()
grouped_target = target_population_df.groupby('일련_번호')
sequences_target = [group[features_for_lstm].values for _, group in tqdm(grouped_target, desc="타겟군 시퀀스 생성 중")]
X_target_AE = pad_sequences_numpy(sequences_target, maxlen=MAX_SEQ_LENGTH)

# 타겟군에 대한 복원 오차 계산
model_AE.eval()
with torch.no_grad():
    target_tensor = torch.from_numpy(X_target_AE).float().to(device)
    reconstructed = model_AE(target_tensor)
    errors = torch.mean((target_tensor - reconstructed)**2, dim=[1, 2]).cpu().numpy()

threshold = np.quantile(errors, 0.95) # 예시로 타겟군 오차의 상위 5%를 의심환자로 간주
print(f"오차 임계값: {threshold:.6f}")

# 임계값을 넘는 환자를 '오진 의심 후보'로 최종 선정
anomaly_indices = np.where(errors > threshold)[0]
AE_misdiagnosis_ids = target_patient_ids[anomaly_indices]
print(f"AE가 탐지한 오진 의심 후보 환자 수: {len(AE_misdiagnosis_ids)}명")


# ==================================================================
#                       PART 2: 분류 검증 모델 (ML)
# ==================================================================
print("\n\n--- PART 2: 분류 모델을 이용한 교차 검증 시작 ---")

# --- 2-1. '분류 전문가' 모델 학습 ---
print("\n--- 2-1. 파킨슨병 vs 뇌졸중 분류 모델 학습 ---")
# 분류 모델 학습 데이터는 "확실한" 단일 진단 환자들로 구성
classifier_df = total_df[total_df['진단_질병_코드'].isin(all_target_codes)].copy()
classifier_df['target'] = np.where(classifier_df['진단_질병_코드'].isin(parkinson_code), 0, 1)

# 환자별 특징 요약 (1인 1행 데이터로 변환)
X_clf = classifier_df.drop(['일련_번호', 'target', '진단_질병_코드'], axis=1)

numeric_cols = X_clf.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['일련_번호', '순번_번호']]
X_clf[numeric_cols] = X_clf[numeric_cols].fillna(0)


# 범주형 컬럼 인코딩
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
categorical_cols = X_clf.select_dtypes(include=['object', 'category']).columns
for col in tqdm(categorical_cols, desc="범주형 피처 인코딩 중"):
    X_clf[col] = LabelEncoder().fit_transform(X_clf[col].astype(str))

# 수치형 데이터 정규화 (Min-Max Scaling)
scaler = MinMaxScaler()
X_clf[numeric_cols] = scaler.fit_transform(X_clf[numeric_cols])
y_clf = classifier_df['target']

from sklearn.model_selection import GroupShuffleSplit
groups = classifier_df['일련_번호']

# GroupShuffleSplit을 사용하여 환자 단위로 데이터를 나눕니다.
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X_clf, y_clf, groups))

X_train_clf = X_clf.iloc[train_idx]
y_train_clf = y_clf.iloc[train_idx]
X_test_clf = X_clf.iloc[test_idx]
y_test_clf = y_clf.iloc[test_idx]

import xgboost as xgb
import catboost as cat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 모델 정의
models = {
    "LightGBM": lgb.LGBMClassifier(objective='binary', random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(random_state=42),
}

# 각 모델별 성능을 저장할 데이터프레임
performance_summary = pd.DataFrame()

for name, model in models.items():
    # 모델 학습
    model.fit(X_train_clf, y_train_clf)
    
    # 예측
    y_pred = model.predict(X_test_clf)
    y_pred_proba = model.predict_proba(X_test_clf)[:, 1]
    
    # 성능 평가
    accuracy = accuracy_score(y_test_clf, y_pred)
    f1 = f1_score(y_test_clf, y_pred)
    auc = roc_auc_score(y_test_clf, y_pred_proba)
    
    # 결과 저장
    performance_summary[name] = [accuracy, f1, auc]
    
# 인덱스 이름 설정
performance_summary.index = ["Accuracy", "F1-Score", "AUC"]

print("\n--- 분류 모델별 성능 요약 ---")
print(performance_summary)

# --- 2-2. AE 후보 목록 교차 검증 ---
print("\n--- 2-2. AE 후보 목록에 대한 2차 소견 요청 ---")
best_model_name = performance_summary.idxmax(axis=1)['F1-Score']
best_model = models[best_model_name]
print(f'{best_model_name}의 성능(f1-score: {performance_summary.loc['F1-Score', best_model_name]:.4f})')

# 최종 결과를 담을 비어있는 데이터프레임 생성
misdiagnosis_candidates = pd.DataFrame()

if len(AE_misdiagnosis_ids) > 0:
    # 1. 앙상블 예측을 위한 데이터 준비
    anomaly_patients_df = total_df[total_df['일련_번호'].isin(AE_misdiagnosis_ids)].copy()
    X_anomaly_val = anomaly_patients_df[X_train_clf.columns]
    
    # 2. 모든 모델로 예측 수행 (앙상블)
    all_predictions = pd.DataFrame(index=X_anomaly_val.index)
    for name, model in models.items():
        predictions = model.predict(X_anomaly_val)
        all_predictions[name] = predictions
    
    # 방문 기록별 최종 예측 (다수결)
    anomaly_patients_df['모델_예측'] = all_predictions.mode(axis=1)[0].astype(int)
    
    # 3. 대표 확률값 계산 (best_model 사용)
    best_model_probs = best_model.predict_proba(X_anomaly_val)
    anomaly_patients_df['뇌졸중일_확률'] = best_model_probs[:, 1]
    
    # 4. '초기 진단' 정보(target)를 anomaly_patients_df에 결합
    target_info = classifier_df[['일련_번호', 'target']].drop_duplicates(subset='일련_번호')
    anomaly_patients_df = pd.merge(anomaly_patients_df, target_info, on='일련_번호', how='left')
    
    # 5. 환자 단위로 결과 종합 (단순화된 핵심 로직)
    patient_level_results = anomaly_patients_df.groupby('일련_번호').agg(
        초기_진단명=('target', lambda x: '파킨슨병' if x.iloc[0] == 0 else '뇌졸중'),
        모델_예측=('모델_예측', lambda x: '파킨슨병' if x.mode()[0] == 0 else '뇌졸중'),
        예측_확률=('뇌졸중일_확률', 'mean') # 환자 방문 기록들의 평균 확률
    ).reset_index()

    # 6. 오진 후보 추출
    misdiagnosis_candidates = patient_level_results[
        patient_level_results['초기_진단명'] != patient_level_results['모델_예측']
    ].copy()

    # 예측 확률을 '모델 예측' 결과에 대한 확률로 변환
    misdiagnosis_candidates['예측_확률'] = np.where(
        misdiagnosis_candidates['모델_예측'] == '뇌졸중',
        misdiagnosis_candidates['예측_확률'],
        1 - misdiagnosis_candidates['예측_확률']
    )
else:
    print("AE가 탐지한 오진 의심 후보가 없습니다.")

print(len(AE_misdiagnosis_ids))
# --- 최종 결과 출력 ---
print("\n--- 최종 검증 결과 ---")
if not misdiagnosis_candidates.empty:
    print("아래 환자들은 AE가 탐지한 후보 중, 초기 진단과 ML 모델 예측이 불일치하는 '오진 의심' 사례입니다:")
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(misdiagnosis_candidates[['일련_번호', '초기_진단명', '모델_예측', '예측_확률']])
else:
    print("AE가 탐지한 후보 중 초기 진단과 ML 모델의 예측이 불일치하는 사례가 없습니다.")

import seaborn as sns
model_AE.eval()
with torch.no_grad():
    train_tensor_vis = torch.from_numpy(X_train_AE).float().to(device)
    reconstructed_normal = model_AE(train_tensor_vis)
    errors_normal = torch.mean((train_tensor_vis - reconstructed_normal)**2, dim=[1, 2]).cpu().numpy()
plt.figure(figsize=(12, 7))
sns.histplot(errors_normal, bins=50, kde=True, color='blue', label='정상군(Normal Group)', stat='density')
sns.histplot(errors, bins=50, kde=True, color='red', label='타겟군(Target Group)', stat='density')

plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'임계값(Threshold): {threshold:.4f}')

# 4. 그래프 제목 및 라벨 설정
plt.title('AE 모델의 복원 오차 분포', fontsize=16)
plt.xlabel('복원 오차 (Reconstruction Error)', fontsize=12)
plt.ylabel('밀도 (Density)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import roc_curve, auc

# --- 시각화 코드 시작 ---


best_model = models[best_model_name]
y_pred_proba = best_model.predict_proba(X_test_clf)[:, 1]

# 2. ROC Curve 계산
fpr, tpr, thresholds = roc_curve(y_test_clf, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 3. ROC Curve 그리기
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance') # 랜덤 분류기 (기준선)

# 4. 그래프 제목 및 라벨 설정
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('거짓 양성 비율 (False Positive Rate)', fontsize=12)
plt.ylabel('진짜 양성 비율 (True Positive Rate)', fontsize=12)
plt.title(f'{best_model_name} 모델 ROC Curve', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

from sklearn.metrics import classification_report

# 가장 성능이 좋았던 모델로 클래스 예측
best_model = models[best_model_name]
y_pred = best_model.predict(X_test_clf)

# 클래스별 성능 리포트 출력
class_names = ['파킨슨병 (Class 0)', '뇌졸중 (Class 1)']
report = classification_report(y_test_clf, y_pred, target_names=class_names)

print("\n--- 분류 성능 상세 보고서 ---")
print(report)

import seaborn as sns
from sklearn.metrics import confusion_matrix

# 혼동 행렬 계산
cm = confusion_matrix(y_test_clf, y_pred)

# Seaborn을 이용해 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

# 그래프 제목 및 라벨 설정
plt.title(f'{best_model_name} 모델 Confusion Matrix', fontsize=16)
plt.xlabel('예측된 레이블 (Predicted Label)', fontsize=12)
plt.ylabel('실제 레이블 (True Label)', fontsize=12)
plt.show()