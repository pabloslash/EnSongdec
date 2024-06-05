import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb


class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, dropout_prob=0.2):
        """
        Args:
            layer_sizes (list of int): The sizes of the layers of the network.
            dropout_prob (float): The dropout probability for regularization.
        """
        super(FeedforwardNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.dropout(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)  
            x = F.elu(x)           
            x = self.dropout(x)    
        x = self.layers[-1](x)    
        return x


# Training loop
def ffnn_train(model, dataloader, optimizer, criterion, num_epochs, val_dataloader=None, log_wandb=True):    
    
    # Send model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()        

    # Initialize empty lists to track losses and errors
    tot_train_loss, tot_train_err = [], []
    tot_val_loss, tot_val_err = [], []
    
    for epoch in range(num_epochs):
            
        train_loss = 0.0
        train_error = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_error += mean_absolute_error(targets, outputs)

        train_error, train_loss = train_error/len(dataloader), train_loss/len(dataloader)
        tot_train_loss.append(train_loss)
        tot_train_err.append(train_error)

        # Evaluate
        if val_dataloader:
            val_loss, val_error = ffnn_evaluate(model, val_dataloader, criterion, device)
            tot_val_loss.append(val_loss)
            tot_val_err.append(val_error)
        else:
            val_loss, val_error = None, None

        # log metrics to wandb
        if log_wandb:
            wandb.log({"train_error": train_error, "train_loss": train_loss, "val_error": val_error, "val_loss": val_loss})
        
        # Print progress every 100 epochs
        if (epoch + 1) % 50 == 0:  
            print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {tot_train_loss[-1]}, Training Error: {tot_train_err[-1]}")
            if val_dataloader: 
                print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {tot_val_loss[-1]}, Validation Error: {tot_val_err[-1]}')
    
    return tot_train_loss, tot_train_err, tot_val_loss, tot_val_err      


# Evaluate
def ffnn_evaluate(model, dataloader, criterion, device):
    
    # Send model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    val_loss = 0.0
    val_error = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            val_error += mean_absolute_error(targets, outputs)

    return val_loss/len(dataloader), val_error/len(dataloader)


# Predict
def ffnn_predict(model, dataloader):
    
    # Send model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predicted_outputs = []
    error = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            error.append(mean_absolute_error(targets, outputs))
            predicted_outputs.append(outputs)
            
    return torch.cat(predicted_outputs, dim=0), error


def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()