import torch
from torch import nn
import matplotlib.pyplot as plt

#this code :
#creatore : @peter
#date : 2026-02-23
#level : simple
#description : this code is a simple implementation of linear regression using pytorch 


class LinearRegression(nn.Module):
    def __init__(this):
        super().__init__()
        this.weight = nn.Parameter(torch.randn(1, dtype=torch.float32))
        this.bias = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


start, end, step = 1, 1000, 0.2
torch.manual_seed(42)


#target value : 
targetW, targetB = 2, 3
def linefunc(x): 
    return targetW * x + targetB







data_x = torch.arange(start, end, step, dtype=torch.float32).unsqueeze(1)
data_y = linefunc(data_x)




split_idx = int(0.8 * len(data_x))
x_train, y_train = data_x[:split_idx], data_y[:split_idx]
x_test, y_test = data_x[split_idx:], data_y[split_idx:]


print(f"Total samples: {len(data_x)}")
print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")




def pplot(x_pred, y_pred, title="Predictions vs Ground Truth"):
    plt.plot(x_train, y_train, c="green", label="Training data")
    plt.plot(x_test, y_test, c="blue", label="Testing data")
    plt.plot(x_pred, y_pred, c="red", label="Predictions")
    plt.title(title, fontsize=16)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()



model = LinearRegression()


with torch.inference_mode():
    y_pred_init = model(x_test)
pplot(x_test, y_pred_init, title="Before Training")






loss_fn = nn.L1Loss()  #you can use L1Loss function nn.L1Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)


track_loss = []

epochs = 60
for epoch in range(epochs):
    model.train()    
    # forward
    y_pred = model(x_train)
    # loss function 
    loss = loss_fn(y_pred, y_train)
    
    #reset gradient 
    optimizer.zero_grad()
    
    # backward propagation
    loss.backward()
    
    #update O (hyperparamters in our case ther is W, B)
    optimizer.step()
    track_loss.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    



model.eval()
with torch.no_grad():
    y_pred_test = model(x_test)



print(f"\nLearned parameters: w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")




for i, (pred, target) in enumerate(zip(y_pred_test[:10], y_test[:10])):
    print(f"Prediction: {pred.item():.3f} -> Target: {target.item():.3f} -> Error: {(pred - target).item():.3f}")

pplot(x_test, y_pred_test, title="After Training")

def LossfuncMSE(y, Y):
    return torch.mean((y - Y) ** 2)

mse_result = LossfuncMSE(y_test, y_pred_test)
print(f"\nMean Squared Error (MSE) on test set: {mse_result.item():.6f}")
from sklearn.metrics import r2_score

with torch.no_grad():

    pre =  model(x_train)
    mse = loss_fn(y_pred, y_train)
    mae = nn.L1Loss()(y_pred, y_train)
    rmse = torch.sqrt(mse)
    r2 = r2_score(y_train.numpy(), y_pred.numpy())
 
print(f"MSE: {mse.item()}")
print(f"MAE: {mae.item()}")
print(f"RMSE: {rmse.item()}")
print(f"R2: {r2}")
# track_loss

#visualaize the error : 


time_line = torch.arange(start=0,end=epochs,step=1)
# pplot(time_line,track_loss)

title = "error curve"
plt.plot(time_line, track_loss, c="red", label="loss curve")

plt.title(title, fontsize=16)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

from pathlib import Path




#model path
ModelPath = Path("models")
ModelPath.mkdir(parents=True,exist_ok=True)

#model_save path 
modelName = "peter_regressionModel01.pth"
modelSavePath = ModelPath/modelName
#save the model in the path
torch.save(obj=model.state_dict,f=modelSavePath)

print(f"model Saved sussufuly ... \n Path : {modelSavePath} ")


