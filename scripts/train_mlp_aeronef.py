import pyLOM
import pyLOM.NN
import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

data =  np.load("data/aeronef/db_random.npy", allow_pickle=True).item()
# data = np.load("data/aeronef/dataset_train.npy") 
x_min, x_max = data['Xcoordinate'].min(axis=1).reshape(-1, 1), data['Xcoordinate'].max(axis=1).reshape(-1, 1)
y_min, y_max = data['Ycoordinate'].min(axis=1).reshape(-1, 1), data['Ycoordinate'].max(axis=1).reshape(-1, 1)
data['Xcoordinate'] = (data['Xcoordinate'] - x_min) / (x_max - x_min)
data['Ycoordinate'] = (data['Ycoordinate'] - y_min) / (y_max - y_min)

x_lim_mask = (data['Xcoordinate'][0] >= 0.2) & (data['Xcoordinate'][0] <= 0.85)
y_lim_mask = (data['Ycoordinate'][0] >= 0.4) & (data['Ycoordinate'][0] <= 0.6)
spatial_mask = x_lim_mask & y_lim_mask

x_coord = data['Xcoordinate']
y_coord = data['Ycoordinate']
spatial_coords = np.stack((x_coord, y_coord), axis=-1)
pressure = data['Pressure']
pressure = pressure[:, spatial_mask]
spatial_coords = spatial_coords[:, spatial_mask, :]
# normalize pressure to zero mean and unit variance
pressure_mean, pressure_std = pressure.mean(), pressure.std()
pressure = (pressure - pressure_mean) / pressure_std
print(pressure.shape, y_coord.shape, spatial_coords.shape)
print(pressure.shape, spatial_coords.shape, y_coord.shape)

vel_inf = data['Vinf']
alpha = data['Alpha']
vel_inf = (vel_inf - vel_inf.min()) / (vel_inf.max() - vel_inf.min())
alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min())
vel_inf = vel_inf.tolist()
alpha = alpha.tolist()

dataset = pyLOM.NN.Dataset(
    variables_out=(pressure,),
    variables_in=spatial_coords[0],
    parameters=(vel_inf, alpha),
    snapshots_by_column=False
)
dataset_train, dataset_test = dataset.get_splits_by_parameters([0.8, 0.2])
sample_input, sample_output = dataset_train[0]
print(sample_input.shape, sample_output.shape)
dataset_train = TensorDataset(dataset_train[:][0], dataset_train[:][1])
dataset_test = TensorDataset(dataset_test[:][0], dataset_test[:][1])
print(len(dataset_train), len(dataset_test))

model = pyLOM.NN.MLP(
    input_size=sample_input.shape[0],
    output_size=sample_output.shape[0],
    hidden_size=128,
    n_layers=4,
    p_dropouts=0.,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

training_params = {
    "epochs": 500,
    "lr": 1e-4,
    "lr_gamma": 0.99,
    "lr_scheduler_step": 100,
    "batch_size": 160000,
    "loss_fn": torch.nn.MSELoss(),
    "optimizer_class": torch.optim.AdamW,
    "print_rate_epoch": 1,
    "print_rate_batch": 15,
    # "pin_memory":False,
    "num_workers":32,
}

pipeline = pyLOM.NN.Pipeline(
    train_dataset=dataset_train,
    test_dataset=dataset_test,
    model=model,
    training_params=training_params,
)

# training_logs = pipeline.run()
# model.save('./aeronef_mlp_model.pt')
model = pyLOM.NN.MLP.load('./aeronef_mlp_model.pt')
preds = model.predict(dataset_test, batch_size=250)
y_true = dataset_test[:][1]
evaluator = pyLOM.NN.RegressionEvaluator()
evaluator(y_true, preds)
evaluator.print_metrics()


def true_vs_pred_plot(y_true, y_pred, path):
    """
    Auxiliary function to plot the true vs predicted values
    """
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=300)

def plot_train_test_loss(train_loss, test_loss, path):
    """
    Auxiliary function to plot the training and test loss
    """
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss) # test loss is calculated at the end of each epoch
    total_iters = len(train_loss) # train loss is calculated at the end of each iteration/batch
    iters_per_epoch = total_iters // total_epochs
    plt.plot(np.arange(iters_per_epoch, total_iters+1, step=iters_per_epoch), test_loss, label="Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)


true_vs_pred_plot(y_true, preds, './true_vs_pred.png')
plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], './train_test_loss.png')