import torch
import torch.nn as nn
import torch.optim as optim

# Definindo a classe da rede neural
class HelloWorldNN(nn.Module):
    def __init__(self):
        super(HelloWorldNN, self).__init__()
        self.fc = nn.Linear(2, 1)  # Camada linear com 2 entradas e 1 saída

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))  # Função de ativação sigmoid para a saída
        return x

# Criando a instância da rede neural
net = HelloWorldNN()

# Definindo os dados de entrada e saída para a operação XOR
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Definindo a função de perda e o otimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Treinamento da rede neural
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, targets)

    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Imprimir o progresso a cada 1000 épocas
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Testando a rede neural treinada
with torch.no_grad():
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predictions = net(test_inputs)
    predictions = (predictions > 0.5).float()  # Convertendo as saídas para 0 ou 1

    print("\nResultados do teste:")
    for i in range(len(test_inputs)):
        print(f'Entrada: {test_inputs[i].tolist()}, Saída da Rede Neural: {predictions[i].tolist()}')
