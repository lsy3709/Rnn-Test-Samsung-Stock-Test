from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import torch.nn as nn
import yfinance as yf
# from flask_cors import CORS
app = Flask(__name__)
# CORS(app)


# 모델 및 스케일러 로드
class StockPredictorRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, output_size=1):
        super(StockPredictorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 모델 로드
model = StockPredictorRNN()
model.load_state_dict(torch.load('./samsungStock.pth', map_location=torch.device('cpu')))
model.eval()

# 스케일러 로드
scaler = torch.load('./scaler.pth', map_location=torch.device('cpu'))

@app.route('/')
def home():
    return render_template('index.html')

# 예측 엔드포인트

# 동적(기간별) 예측
@app.route('/predict2', methods=['POST'])
def predict2():
    try:
        # JSON 데이터에서 입력 값 추출
        data = request.get_json()
        if not data or 'data' not in data or 'period' not in data:
            print("데이터가 수신되지 않았거나 유효하지 않습니다.")
            return jsonify({"error": "요청에 데이터 또는 기간 정보가 포함되어 있지 않습니다."}), 400

        print("수신된 데이터:", data)  # Debug print to check received data
        input_data = data['data']
        period = data['period']

        # 기간에 따른 데이터 길이 정의
        # 당일 꺼로 확인시, 하루씩 빼기
        period_days_map = {
            '1d': 1,
            '5d': 4,
            '1mo': 22,
            '3mo': 59,
            '6mo': 123,
            '1y': 244
        }

        if period not in period_days_map:
            print("지원되지 않는 기간입니다:", period)
            return jsonify({"error": "지원되지 않는 기간입니다."}), 400

        expected_length = period_days_map[period]

        # 입력 데이터가 올바른지 확인
        if not isinstance(input_data, list) or len(input_data) != expected_length:
            print(f"잘못된 입력입니다. {expected_length}일치 데이터가 필요합니다.")
            return jsonify({"error": f"잘못된 입력입니다. {expected_length}일치 Open, High, Low, Close 데이터를 제공하세요."}), 400

        # 이후 로직 처리...
        print("처리할 데이터:", input_data)

        # 입력된 데이터를 numpy 배열로 변환 후 스케일링
        input_data = np.array(input_data)  # 입력 데이터 배열 생성
        input_data = scaler.transform(input_data)  # 스케일러로 정규화
        input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
        input_data = torch.Tensor(input_data)

        # 예측 수행
        with torch.no_grad():
            prediction = model(input_data).item()

        # 예측 결과를 종가 기준으로 역정규화
        prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]

        # 예측 결과 및 확인 메시지 출력
        print(f"예측된 값 (종가): {round(prediction, 2)}")

        # 예측 결과 반환
        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500


# 요청 일수에 따라 동적으로 받기
@app.route('/get_stock_data2', methods=['GET'])
def get_stock_data2():
    # 클라이언트가 요청한 기간
    period = request.args.get('period', default='5d')  # 기본값은 '5d'

    # 삼성전자 종목 코드: '005930.KS' (야후 파이낸스 형식)
    ticker = '005930.KS'
    # 요청된 기간의 데이터를 가져오기
    data = yf.download(ticker, period=period, interval='1d')

    # 데이터 열 이름 출력 (확인용)
    print("데이터 열 이름:", data.columns)
    print("데이터 size:", data.size)

    # MultiIndex가 설정된 경우 첫 번째 레벨만 선택하여 열 이름을 단순화
    data.columns = data.columns.get_level_values(0)

    # 요청된 기간의 데이터 처리
    if period == '1d':
        # 1일의 경우 그대로 반환
        data_subset = data[['Open', 'Low', 'High', 'Close']]
    else:
        # 나머지 기간의 경우 최근 1일을 제외하고 반환
        data_subset = data.iloc[:-1][['Open', 'Low', 'High', 'Close']]

    data_subset = data_subset.reset_index()  # Date 인덱스를 컬럼으로 변환
    data_subset['Date'] = data_subset['Date'].astype(str)  # Date 열을 문자열로 변환

    # DataFrame을 JSON으로 변환 가능한 딕셔너리로 변환
    stock_data = data_subset.to_dict(orient='records')

    return jsonify(stock_data)

if __name__ == '__main__':
    app.run()
