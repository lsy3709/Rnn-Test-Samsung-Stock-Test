from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import torch.nn as nn
import yfinance as yf
app = Flask(__name__)

# 모델 및 스케일러 로드
class StockPredictorRNN(nn.Module):  # 간단한 RNN 기반 주식 예측 모델 정의
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, output_size=1):
        super(StockPredictorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # 순전파 정의
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 은닉 상태 정의
        out, _ = self.rnn(x, h0)  # RNN 레이어로 데이터 처리, out 모양 예시: [배치 크기, 시퀀스 길이, 은닉 크기] (예: [64, 60, 50])
        out = self.fc(out[:, -1, :])  # 마지막 시퀀스의 은닉 상태로 최종 출력값 생성, out 모양 예시: [배치 크기, 1] (예: [64, 1])
        return out

# 모델 로드
model = StockPredictorRNN()  # 모델 인스턴스 생성
model.load_state_dict(torch.load('./samsungStock.pth', map_location=torch.device('cpu')))  # 모델 가중치 로드
model.eval()  # 평가 모드로 설정

# 스케일러 로드
scaler = torch.load('./scaler.pth', map_location=torch.device('cpu'))  # 데이터 스케일러 로드

#======================================================================================================================
# 추가1: LSTM 모델 정의
class LSTMModel(nn.Module):  # PyTorch의 LSTM 모델 클래스 정의
    def __init__(self, input_size=4, hidden_size=128, output_size=1, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()  # nn.Module의 생성자 호출
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)  # LSTM 레이어 정의
        self.fc = nn.Linear(hidden_size, output_size)  # 완전 연결 레이어 정의
        self.relu = nn.ReLU()  # 활성화 함수 ReLU 정의

    def forward(self, x):  # 순전파 함수 정의
        lstm_out, _ = self.lstm(x)  # LSTM의 출력 계산, lstm_out 모양 예시: [배치 크기, 시퀀스 길이, 은닉 크기] (예: [64, 60, 128])
        last_out = lstm_out[:, -1, :]  # 마지막 시퀀스의 출력을 선택, last_out 모양 예시: [배치 크기, 은닉 크기] (예: [64, 128])
        out = self.fc(self.relu(last_out))  # ReLU 활성화 후 완전 연결 레이어 통과, out 모양 예시: [배치 크기, 출력 크기] (예: [64, 1])
        return out

# LSTM 모델 로드
model2 = LSTMModel()  # LSTM 모델 인스턴스 생성
model2.load_state_dict(torch.load('./samsungStock_LSTM_60days_basic.pth', map_location=torch.device('cpu')))  # 모델 가중치 로드
model2.eval()  # 평가 모드로 설정
scaler = torch.load('./scaler_LSTM_60days_basic.pth')  # 스케일러 로드

#======================================================================================================================
# 추가2: GRU 모델 정의
class GRUModel(nn.Module):  # PyTorch를 사용하여 GRU(Gated Recurrent Unit) 모델을 정의합니다
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
        super(GRUModel, self).__init__()  # nn.Module의 생성자 호출
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU 레이어 정의
        self.fc = nn.Linear(hidden_size, output_size)  # 완전 연결 레이어 정의

    def forward(self, x):  # 순전파 정의
        out, _ = self.gru(x)  # GRU 레이어를 통해 입력 처리, out 모양 예시: [배치 크기, 시퀀스 길이, 은닉 크기] (예: [64, 60, 64])
        out = self.fc(out[:, -1])  # 마지막 시퀀스 은닉 상태로 선형 레이어에 전달, out 모양 예시: [배치 크기, 출력 크기] (예: [64, 1])
        return out

# GRU 모델 로드
model3 = GRUModel()  # GRU 모델 인스턴스 생성
model3.load_state_dict(torch.load('./samsungStock_GRU.pth', map_location=torch.device('cpu')))  # 모델 가중치 로드
model3.eval()  # 평가 모드로 설정
scaler = torch.load('./scaler_GRU.pth')  # 스케일러 로드

@app.route('/')
def home():  # 홈 페이지 라우트 설정
    return render_template('index.html')

# 예측 엔드포인트
@app.route('/predict1', methods=['POST'])
def predict2():  # RNN 모델 예측 엔드포인트
    try:
        data = request.get_json()  # JSON 데이터에서 입력 값 추출
        if not data or 'data' not in data or 'period' not in data:  # 데이터 검증
            return jsonify({"error": "요청에 데이터 또는 기간 정보가 포함되어 있지 않습니다."}), 400
        input_data = data['data']
        period = data['period']

        period_days_map = {
            '1d': 1,
            '5d': 4,
            '1mo': 22,
            '3mo': 59,
            '6mo': 123,
            '1y': 244
        }

        if period not in period_days_map:
            return jsonify({"error": "지원되지 않는 기간입니다."}), 400

        expected_length = period_days_map[period]

        if not isinstance(input_data, list) or len(input_data) != expected_length:
            return jsonify({"error": f"잘못된 입력입니다. {expected_length}일치 Open, High, Low, Close 데이터를 제공하세요."}), 400

        input_data = np.array(input_data)  # 입력 데이터 배열 생성
        input_data = scaler.transform(input_data)  # 스케일러로 정규화
        input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
        input_data = torch.Tensor(input_data)

        with torch.no_grad():
            prediction = model(input_data).item()  # RNN 모델 예측 수행

        prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]  # 종가 기준으로 역정규화

        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500

# LSTM 예측 엔드포인트
@app.route('/predict2', methods=['POST'])
def predict3():
    try:
        data = request.get_json()  # JSON 데이터에서 입력 값 추출
        if not data or 'data' not in data or 'period' not in data:
            return jsonify({"error": "요청에 데이터 또는 기간 정보가 포함되어 있지 않습니다."}), 400
        input_data = data['data']
        period = data['period']

        period_days_map = {
            '1d': 1,
            '5d': 4,
            '1mo': 22,
            '3mo': 59,
            '6mo': 123,
            '1y': 244
        }

        if period not in period_days_map:
            return jsonify({"error": "지원되지 않는 기간입니다."}), 400

        expected_length = period_days_map[period]

        if not isinstance(input_data, list) or len(input_data) != expected_length:
            return jsonify({"error": f"잘못된 입력입니다. {expected_length}일치 Open, High, Low, Close 데이터를 제공하세요."}), 400

        input_data = np.array(input_data)  # 입력 데이터 배열 생성
        input_data = scaler.transform(input_data)  # 스케일러로 정규화
        input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
        input_data = torch.Tensor(input_data)

        with torch.no_grad():
            prediction = model2(input_data).item()  # LSTM 모델 예측 수행

        prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]  # 종가 기준으로 역정규화

        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500

# GRU 예측 엔드포인트
@app.route('/predict3', methods=['POST'])
def predict4():
    try:
        data = request.get_json()  # JSON 데이터에서 입력 값 추출
        if not data or 'data' not in data or 'period' not in data:
            return jsonify({"error": "요청에 데이터 또는 기간 정보가 포함되어 있지 않습니다."}), 400
        input_data = data['data']
        period = data['period']

        period_days_map = {
            '1d': 1,
            '5d': 4,
            '1mo': 22,
            '3mo': 59,
            '6mo': 123,
            '1y': 244
        }

        if period not in period_days_map:
            return jsonify({"error": "지원되지 않는 기간입니다."}), 400

        expected_length = period_days_map[period]

        if not isinstance(input_data, list) or len(input_data) != expected_length:
            return jsonify({"error": f"잘못된 입력입니다. {expected_length}일치 Open, High, Low, Close 데이터를 제공하세요."}), 400

        input_data = np.array(input_data)  # 입력 데이터 배열 생성
        input_data = scaler.transform(input_data)  # 스케일러로 정규화
        input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
        input_data = torch.Tensor(input_data)

        with torch.no_grad():
            prediction = model3(input_data).item()  # GRU 모델 예측 수행

        prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]  # 종가 기준으로 역정규화

        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500

# 요청 일수에 따라 동적으로 받기
@app.route('/get_stock_data2', methods=['GET'])
def get_stock_data2():
    period = request.args.get('period', default='5d')  # 기본 요청 기간 설정
    ticker = '005930.KS'  # 삼성전자 종목 코드
    data = yf.download(ticker, period=period, interval='1d')  # 지정된 기간 동안 주식 데이터 가져오기

    data.columns = data.columns.get_level_values(0)  # MultiIndex가 설정된 경우 열 이름 단순화

    if period == '1d':
        data_subset = data[['Open', 'Low', 'High', 'Close']]  # 1일의 경우 데이터를 그대로 반환
    else:
        data_subset = data.iloc[:-1][['Open', 'Low', 'High', 'Close']]  # 나머지 기간은 최근 1일 제외하고 반환

    data_subset = data_subset.reset_index()  # Date 인덱스를 컬럼으로 변환
    data_subset['Date'] = data_subset['Date'].astype(str)  # Date를 문자열로 변환

    stock_data = data_subset.to_dict(orient='records')  # JSON으로 변환 가능한 딕셔너리로 변환

    return jsonify(stock_data)

if __name__ == '__main__':
    app.run()
