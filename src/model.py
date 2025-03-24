import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_percentage_error
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    # Ensure no division by zero by adding a small epsilon value
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    smape = np.mean( np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return smape

def median_absolute_percentage_error(true_values, predicted_values):
    """
    Calculate the Median Absolute Percentage Error (MedAPE).

    Parameters:
    - true_values (array-like): Array of true values.
    - predicted_values (array-like): Array of predicted values.
    - epsilon (float): Small value to avoid division by zero errors (default is 1e-8).

    Returns:
    - medape (float): The Median Absolute Percentage Error as a percentage.
    """
    # Convert inputs to numpy arrays for calculation
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # Calculate absolute percentage errors, adding epsilon to avoid division by zero
    absolute_percentage_errors = np.abs((true_values - predicted_values) / (true_values+1e-6))

    # Calculate MedAPE
    medape = np.median(absolute_percentage_errors)

    return medape

def mean_absolute_percentage_error2(true_values, predicted_values):
    """
    Calculate the Median Absolute Percentage Error (MedAPE).

    Parameters:
    - true_values (array-like): Array of true values.
    - predicted_values (array-like): Array of predicted values.
    - epsilon (float): Small value to avoid division by zero errors (default is 1e-8).

    Returns:
    - medape (float): The Median Absolute Percentage Error as a percentage.
    """
    # Convert inputs to numpy arrays for calculation
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # Calculate absolute percentage errors, adding epsilon to avoid division by zero
    absolute_percentage_errors = np.abs((true_values - predicted_values) / (true_values+1e-4))
 
    # Calculate MedAPE
    medape = np.mean(absolute_percentage_errors)

    return medape

## Optimized pure MLP architecture for ADM
class MLP(nn.Module):
  def __init__(self, l, n, d_i, d_o):
    super(MLP, self).__init__()
    self.linear = [n for i in range(l)]
    self.linear[0] = d_i
    self.linear[-1] = d_o
    self.linears = nn.ModuleList([])
    self.bn_linears = nn.ModuleList([])

    for ind, fc_num in enumerate(self.linear[:-1]):
      self.linears.append(nn.Linear(fc_num, self.linear[ind + 1]))
      self.bn_linears.append(nn.BatchNorm1d(self.linear[ind + 1]))

  def forward(self, g):
    out = g
    for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
      if ind < len(self.linears) - 1:
        # print(ind)
        # print(fc(out).shape)
        out = F.relu(bn(fc(out)))
      else:
        out = fc(out)

    return out
  
  def train_model(self, device, trainloader, testloader, Loss, optimizer, scheduler=None, num_epochs=100, eval_step=10):
    train_loss_arr = []
    val_loss_arr = []
    best_val_loss = torch.inf

    for epoch in range(num_epochs):
      train_running_loss = 0.0
      val_running_loss = 0
      self.train()

      for i, (g, s) in enumerate(trainloader):
        g = g.to(device)
        s = s.to(device)

        logits = self(g)
        loss = Loss(logits, s)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_running_loss += loss.detach().item()
        del loss, logits

      train_avg_loss = train_running_loss/(i+1)

      self.eval()
      if epoch%eval_step == 0:
        with torch.no_grad():
          for j, (g, s) in enumerate(testloader):
            g = g.to(device)
            s = s.to(device)

            logits = self(g)
            loss = Loss(logits, s)
            val_running_loss += loss.detach().item()
            del loss, logits

        val_avg_loss = val_running_loss/(j+1)
        print('Epoch: %d | Train Loss: %.4f | Test Loss: %.4f'%(epoch, train_avg_loss, val_avg_loss))
        train_loss_arr.append(train_avg_loss)
        val_loss_arr.append(val_avg_loss)

      if val_avg_loss < best_val_loss:
          best_val_loss = val_avg_loss
          torch.save(self.state_dict(), 'model-best.pth')
          print(f'Epoch {epoch}: Lower validation loss observed - saving model')

      if scheduler != None:
        scheduler.step(train_avg_loss)

    return train_loss_arr, val_loss_arr
  
  
  def validate_model(self, device, valloader, Loss, index):
    mse_val_arr = np.zeros(len(valloader))
    smare_val_arr = np.zeros(len(valloader))
    mare_val_arr = np.zeros(len(valloader))
    medare_val_arr = np.zeros(len(valloader))
    s_trues = []
    s_preds = []


    self.eval()
    with torch.no_grad():
      for i, (g, s) in enumerate(valloader):
        s_trues.append(s.cpu().numpy())

        g = g.to(device)
        s = s.to(device)

        logits = self(g)
        # print(logits.cpu().numpy())
        loss = Loss(logits, s)
        mse_val_arr[i] = loss.detach().cpu().numpy()
        # print(type(logits.cpu().numpy())) 
        epsilon = 1e-6
        s_pred = logits.cpu().numpy()
        s_preds.append(s_pred)
        # s_pred = np.round(s_pred, decimals=3)
        smare_val_arr[i] = symmetric_mean_absolute_percentage_error(s.cpu().numpy(),s_pred)
        medare_val_arr[i] = median_absolute_percentage_error(s.cpu().numpy(),s_pred)
        mare_val_arr[i] = mean_absolute_percentage_error( s.cpu().numpy(),s_pred)
        del loss, logits
    np.save(f'NN_trues_{index}.npy', s_trues)
    np.save(f'NN_preds_{index}.npy', s_preds)
    return mse_val_arr, smare_val_arr, medare_val_arr, mare_val_arr
  

  def initialize_from_uniform_to_dataset_distrib(self, geometry_eval):
    """
    since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
    to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
    of the X prior given in the training set and data generation process
    :param geometry_eval: The input uniform distribution from [0,1]
    :return: The transformed initial guess from prior distribution
    """
    X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
    geometry_eval_input = geometry_eval * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
    if self.flags.data_set == 'robotic_arm' or self.flags.data_set == 'ballistics':
        return geometry_eval
    return geometry_eval_input
    #return geometry_eval

  def get_boundary_lower_bound_uper_bound(self):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        if self.flags.data_set == 'sine_wave': 
            return np.array([2, 2]), np.array([-1, -1]), np.array([1, 1])
        elif self.flags.data_set == 'meta_material':
            return np.array([2.272,2.272,2.272,2.272,2,2,2,2]), np.array([-1,-1,-1,-1,-1,-1,-1,-1]), np.array([1.272,1.272,1.272,1.272,1,1,1,1])
       



  def initialize_geometry_eval(self):
      """
      Initialize the geometry eval according to different dataset. These 2 need different handling
      :return: The initialized geometry eval
      """
      geomtry_eval = torch.rand([self.flags.eval_batch_size, 14], requires_grad=True, device='cuda')
      #geomtry_eval = torch.randn([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
      return geomtry_eval


  def make_optimizer_eval(self, geometry_eval, optimizer_type=None):
      """
      The function to make the optimizer during evaluation time.
      The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
      :return: the optimizer_eval
      """
      if optimizer_type is None:
          optimizer_type = self.flags.optim
      if optimizer_type == 'Adam':
          op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
      elif optimizer_type == 'RMSprop':
          op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
      elif optimizer_type == 'SGD':
          op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
      else:
          raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
      return op

  def make_lr_scheduler(self, optm):
      """
      Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
      1. ReduceLROnPlateau (decrease lr when validation error stops improving
      :return:
      """
      return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                            factor=self.flags.lr_decay_rate,
                                            patience=10, verbose=True, threshold=1e-4)



  def evaluate_one(self, target_spectra, save_dir='data/', MSE_Simulator=False ,save_all=False, ind=None, save_misc=False, save_Simulator_Ypred=False):
    """
    The function which being called during evaluation and evaluates one target y using # different trails
    :param target_spectra: The target spectra/y to backprop to 
    :param save_dir: The directory to save to when save_all flag is true
    :param MSE_Simulator: Use Simulator Loss to get the best instead of the default NN output logit
    :param save_all: The multi_evaluation where each trail is monitored (instad of the best) during backpropagation
    :param ind: The index of this target_spectra in the batch
    :param save_misc: The flag to print misc information for degbugging purposes, usually printed to best_mse
    :return: Xpred_best: The 1 single best Xpred corresponds to the best Ypred that is being backproped 
    :return: Ypred_best: The 1 singe best Ypred that is reached by backprop
    :return: MSE_list: The list of MSE at the last stage
    """

    # Initialize the geometry_eval or the initial guess xs
    geometry_eval = self.initialize_geometry_eval()
    # Set up the learning schedule and optimizer
    self.optm_eval = self.make_optimizer_eval(geometry_eval)#, optimizer_type='SGD')
    self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
    
    # expand the target spectra to eval batch size
    target_spectra_expand = target_spectra.expand([self.flags.eval_batch_size, -1])

    # Begin NA
    for i in range(self.flags.backprop_step):
        # Make the initialization from [-1, 1], can only be in loop due to gradient calculator constraint
        geometry_eval_input = self.initialize_from_uniform_to_dataset_distrib(geometry_eval)
        if save_misc and ind == 0 and i == 0:                       # save the modified initial guess to verify distribution
            np.savetxt('geometry_initialization.csv',geometry_eval_input.cpu().data.numpy())
        self.optm_eval.zero_grad()                                  # Zero the gradient first
        logit = self.model(geometry_eval_input)                     # Get the output
        ###################################################
        # Boundar loss controled here: with Boundary Loss #
        ###################################################
        loss = self.make_loss(logit, target_spectra_expand, G=geometry_eval_input)         # Get the loss
        loss.backward()                                             # Calculate the Gradient
        # update weights and learning rate scheduler
        if i != self.flags.backprop_step - 1:
            self.optm_eval.step()  # Move one step the optimizer
            self.lr_scheduler.step(loss.data)
    
    if save_all:                # If saving all the results together instead of the first one
        ##############################################################
        # Choose the top "trail_nums" points from NA solutions #
        ##############################################################
        mse_loss = np.reshape(np.sum(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1), [-1, 1])
        mse_loss = np.concatenate((mse_loss, np.reshape(np.arange(self.flags.eval_batch_size), [-1, 1])), axis=1)
        loss_sort = mse_loss[mse_loss[:, 0].argsort(kind='mergesort')]                         # Sort the loss list
        exclude_top = 0
        trail_nums = 1000
        good_index = loss_sort[exclude_top:trail_nums+exclude_top, 1].astype('int')                        # Get the indexs
        print("In save all funciton, the top 10 index is:", good_index[:10])
        saved_model_str = self.saved_model.replace('/', '_') + 'inference' + str(ind)
        Ypred_file = os.path.join(save_dir, 'test_Ypred_point{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_point{}.csv'.format(saved_model_str))
        if self.flags.data_set != 'meta_material':  # This is for meta-meterial dataset, since it does not have a simple simulator
            # 2 options: simulator/logit
            Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
            if not save_Simulator_Ypred:            # The default is the simulator Ypred output
                Ypred = logit.cpu().data.numpy()
            if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred, [-1, 1])
            with open(Xpred_file, 'a') as fxp, open(Ypred_file, 'a') as fyp:
                np.savetxt(fyp, Ypred[good_index, :])
                np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index, :])
        else:
            with open(Xpred_file, 'a') as fxp:
                np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index, :])
    ###################################
    # From candidates choose the best #
    ###################################
    Ypred = logit.cpu().data.numpy()

    if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
        Ypred = np.reshape(Ypred, [-1, 1])
    
    # calculate the MSE list and get the best one
    MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
    best_estimate_index = np.argmin(MSE_list)
    print("The best performing one is:", best_estimate_index)
    Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
    if save_Simulator_Ypred:
        Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
        if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
            Ypred = np.reshape(Ypred, [-1, 1])
    Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

    return Xpred_best, Ypred_best, MSE_list


