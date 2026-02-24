#creator : @peter 
#type : simple
#obj : this code present another method u can use for linear regression in
#the class extended from nn.Module
#so we dont need to hundle paramaters manuly but we can use just 
#predefined function from pytorch.Linear(...)

#import lib

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path



#creat our data :

START = 0
END = 150
STEP = 0.5
W = 2
B = 3
SEED_KERNAL = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#linear function

def linear_fun(x:torch.tensor)->torch.tensor:
  return W*x+B


torch.manual_seed(seed=SEED_KERNAL)
data = torch.arange(start=START,end=END,step=STEP)
data.to(device=DEVICE)
data = data.unsqueeze(dim=1)


# data_size = data.size()
# data_size = data.shape
data_size = len(data)
split_point =int( data_size * 0.8)


#split the data
# training : 0.8
# testing : 0.2

print(data_size,split_point)

x_training = data[:split_point]
y_training = linear_fun(x_training)
x_testing = data[split_point:]
y_testing = linear_fun(x_testing)

x_training = x_training.to(DEVICE)
y_training = y_training.to(DEVICE)
x_testing  = x_testing.to(DEVICE)
y_testing  = y_testing.to(DEVICE)




# print(x_training[:10],y_training[:10])
# print(len(x_training)==len(y_training))
# print(len(x_testing)==len(y_testing))





class LinearRegression(nn.Module):

  def __init__(this):
    super().__init__()
    # this.Weight = nn.Parameter(randn(1,type=torch.float64),requires_grad=True)
    # this.Bais = nn.Parameter(randn(1,type=torch.float64),requires_grad=True)
    this.LR = nn.Linear(in_features=1,out_features=1,bias=True)



  def forward(this,x:torch.tensor)->torch.tensor:
    # return this.Weight * x + this.Bais
    return this.LR(x)

learning_rate = 0.001


torch.manual_seed(seed=SEED_KERNAL)
mymodel = LinearRegression()
mymodel.to(device=DEVICE)
# list(mymodel.parameters())
#loss function :
lossFn = nn.L1Loss()

#optimaizer :

optimaizer = torch.optim.SGD(mymodel.parameters(),lr=learning_rate)



epochs = 100
track_loss=[]



for epoc in range(epochs):
  #training mode :
  mymodel.train()
  #predict
  y_predict = mymodel(x_training)
  #calculate the loss
  loss = lossFn(y_predict,y_training)
  track_loss.append(loss.item())
  #reset gradient for no ++ in them
  optimaizer.zero_grad()

  #backward : calculate gradient
  loss.backward()

  #update Q
  optimaizer.step()

  if(epoc % 10 == 0): 
    print(f"trainign test : epoch {epoc} -> loss : {loss.item()} ")
    mymodel.eval()
    with torch.inference_mode():
      y_prediction = mymodel(x_testing)
      cost = lossFn(y_prediction,y_testing)
      print(f"validation test : epoch {epoc} -> loss : {loss.item()} ")
  
  # print("\n************************\n")



time_line = torch.arange(start=0,end=epochs//2,step=1/2)

print(len(time_line) , len(track_loss))

title = "error curve"
plt.plot(time_line, track_loss, c="red", label="loss curve")



plt.title(title, fontsize=16)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()


title = "testing the model"
plt.plot(x_testing.cpu().numpy(),y_testing.cpu().numpy(), c="green", label="actual")
plt.plot(x_testing.cpu().numpy(),y_prediction.cpu().numpy(), c="red", label="predict")


plt.title(title, fontsize=16)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()




# #save and load the model :


# model_path = Path("models")
# model_path.mkdir(parents=True,exist_ok=True)


# model_name = "mymodel.pth"
# MODELPATH = model_path/model_name


# #save the model

# torch.save(obj=mymodel.state_dict(),f=MODELPATH)




# modelv1.eval()
# with torch.inference_mode():
#   _mytensor = torch.tensor([0.1,0.2,0.3,0.4],dtype=torch.float32).to(device=DEVICE).unsqueeze(dim=1)
#   result = linear_fun(_mytensor)
#   newpredict = modelv1(_mytensor)
#   lossr = lossFn(newpredict,result)

# print(f"the loss of this data is : {lossr}")


