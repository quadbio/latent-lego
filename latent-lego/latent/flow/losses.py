from keras import losses

def mse():
   def mse_loss(y_true, y_pred):
      return losses.mean_squared_error(y_true, y_pred)
   return mse_loss
