from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import torch.nn as nn
app = Flask(__name__)


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
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON 데이터에서 입력 값 추출
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"error": "요청에 데이터가 포함되어 있지 않습니다."}), 400

        input_data = data['data']

        # 입력 데이터가 올바른지 확인 (2일치 데이터 확인)
        if not isinstance(input_data, list) or len(input_data) != 2:
            return jsonify({"error": "잘못된 입력입니다. 2일치 Open, High, Low, Close 데이터를 제공하세요."}), 400

        # 입력된 2일치 데이터를 numpy 배열로 변환 후 스케일링
        input_data = np.array(input_data)  # 2일간의 입력 데이터 배열 생성
        input_data = scaler.transform(input_data)  # 스케일러로 정규화
        input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
        input_data = torch.Tensor(input_data)

        # 예측 수행
        with torch.no_grad():
            prediction = model(input_data).item()

        # 예측 결과를 종가 기준으로 역정규화
        prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]

        # 예측 결과 반환
        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        # 예외가 발생할 경우 JSON으로 에러 반환
        return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500


if __name__ == '__main__':
    app.run()
