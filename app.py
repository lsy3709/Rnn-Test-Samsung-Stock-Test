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
    # 사용자 입력 값 받기
    data = request.form
    open_price = float(data['Open'])
    high_price = float(data['High'])
    low_price = float(data['Low'])
    close_price = float(data['Close'])

    # 데이터 정규화 및 텐서화
    input_data = np.array([[open_price, high_price, low_price, close_price]])
    input_data = scaler.transform(input_data)  # 정규화
    input_data = torch.Tensor(input_data).unsqueeze(0)  # 배치 차원 추가

    # 예측
    with torch.no_grad():
        prediction = model(input_data).item()
    prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]  # Close 역정규화

    return jsonify({'Predicted Close': round(prediction, 2)})

if __name__ == '__main__':
    app.run()
