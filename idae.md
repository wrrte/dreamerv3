termination도 예측함? 그렇다면 termination 때는 그 예측 loss 가중치도 높게

그리고 termination 확률이 높을 때도 termination loss와 reconstrcuction loss 가중치를 높게. 
termination loss 가중치는 지수 형태로 하는 건 어떰? 
termination에 가까울수록 높아지고, 에피소드 길이에 비례하여 반감기를 정하고. 
안전을 위해 초반에는 반감기를 최대한 크게 세팅해두고 반감기는 에피소드 길이를 ema로 측정하여 갱신하고.




일단 vae에 L1을 구현할 거임. 현재 성능이 급락했는데 이유 분석해보기. 그다음 GPT 의견대로 해보자.