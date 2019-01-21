class FC_Clf(nn.Module):

    def __init__(self, num_feature, num_class):
        super(Multiclass_Classification, self).__init__()

        # 이전 예제에서는 __init__에 self만 있었습니다.
        # 이제는 넣고 싶은 feature의 개수와 class를 자유롭게 하기 위해 변수로 지정합니다.
        # Layer 역시 활성화 함수를 ReLU로 변경하고 Layer 안에 넣어 좀 더 짧은 코드로 신경망을 구성했습니다.

        self.Layer_1 = nn.Sequential(
                                nn.Linear(num_feature, 100),
                                nn.ReLU()
                            )
        self.Layer_2 = nn.Sequential(
                                nn.Linear(100, 50),
                                nn.ReLU()
                            )

        self.Layer_3 = nn.Sequential(
                                nn.Linear(50, 30),
                                nn.ReLU()
                            )

        self.dropout = nn.Dropout(0.5)

        self.out = nn.Linear(30, num_class)

    def forward(self, inputs):

        x = self.Layer_1(inputs)
        x = self.Layer_2(x)
        x = self.dropout(x)
        x = self.Layer_3(x)
        x = self.dropout(x)

        x = self.out(x)

        return x

    def predict(self, test_inputs):

        x = self.Layer_1(inputs)
        x = self.Layer_2(x)
        x = self.Layer_3(x)

        x = self.out(x)

        # 1차원의 최대값 위치들을 출력합니다.

        return torch.max(x, 1)[1]
