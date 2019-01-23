def train():

    EPOCHS = parameter['epochs']
    BATCH_SIZE = parameter['batch_size']
    NUM_FEATURE = parameter['num_feature']
    NUM_CLASS = parameter['num_class']

    model = FC_Clf(NUM_FEATURE, NUM_CLASS)

    # pytorch에서는 NLL loss + log softmax가 이미 적용되어 있습니다.
    criterion = nn.CrossEntropyLoss()
    iris = load_iris()
    train_data = get_train_data(iris,TEST_SIZE)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):

        for X_batch, y_batch in train_loader:

            #torch.tensor는 기본적으로 float로 들어갑니다.

            inputs = torch.Tensor(X_batch.float())
            targets = torch.LongTensor(y_batch.long())
            model.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(loss)
    torch.save
