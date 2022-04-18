import time
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import deepwave
from  scipy.ndimage import gaussian_filter, gaussian_filter1d
import horovod.torch as hvd


class fwi():
    def __init__(self,par,acquisition):
       self.nx=par['nx']
       self.nz=par['nz']
       self.dx=par['dx']
       self.nt=par['nt']
       self.dt=par['dt']
       self.num_dims=par['num_dims']
       self.num_shots=par['ns']
       self.num_batches=par['num_batches']
       self.num_sources_per_shot=1
       self.num_receivers_per_shot = par['nr']
       self.ds= par['ds']
       self.dr= par['dr']
       self.sz = par['sz']
       self.rz = par['rz']
       self.os = par['osou']
       self.orec = par['orec']
       self.ox = par['ox']
       self.num_sources_per_shot=1

        
       self.s_cor, self.r_cor =self.get_coordinate(acquisition)


    def get_coordinate(self,mode):
       """ 
       Create arrays containing the source and receiver locations
       ------------------------------------------------------------
       Argumunts :
       ----------
       mode:  int: 1 or 2
           1: Offset is not specified and the recievers are the same for all shots 
           2: Recivers are designed to have specific offset (p.s. this should be done from the parameters) 
       
       output: 
       -------
        x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
        x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
        Note: the depth is set to zero , to change it change the first element in the last dimensino 
        """
       x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
       x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
       # x direction 
       x_s[:, 0, 1] = torch.arange(0,0+self.num_shots * self.ds , self.ds).float() + self.os  
       # z direction  
       x_s[:, 0, 0] = self.sz
                        
       if mode ==1:
         # x direction 
         x_r[0, :, 1] = torch.arange(0,self.num_receivers_per_shot).float() * self.dr + self.orec
         x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
         # z direction 
         x_r[:, :, 0] = self.rz
       elif mode ==2: # fixed spread !! 
         # x direction 
         # for i in range (self.num_shots):
         #    orec =  i * self.ds
         #    x_r[i, :, 1] = torch.arange(0,self.num_receivers_per_shot* self.dr,self.dr).float() + orec 
         x_r[0,:,1] = torch.arange(self.num_receivers_per_shot).float() * self.dr   + self.orec
         x_r[:,:,1] = x_r[0,:,1].repeat(self.num_shots,1) + \
                  torch.arange(0,self.num_shots).repeat(self.num_receivers_per_shot,1).T * self.ds
         # z direction 
         x_r[:, :, 0] = self.rz

         # Avoid out-of-bound error   
         xr_corr = x_r[:,:,1]
         xr_corr [xr_corr < 0 ] = 0
         xr_corr [xr_corr > (self.nx-2)*self.dx ] = (self.nx-2)*self.dx # ~last point in the model 
         x_r[:, :, 1] =  xr_corr
        
        
 
       return x_s,x_r


    def Ricker(self,freq):
        wavelet = (deepwave.wavelets.ricker(freq, self.nt, self.dt, 1/freq)
                                 .reshape(-1, 1, 1))

                        
        return wavelet

    def forward_modelling(self,model,wavelet,device):
       # pml_width parameter control the boundry, for free surface first argument should be 0 
       prop = deepwave.scalar.Propagator({'vp': model.to(device)}, self.dx,pml_width=[0,20,20,20,20,20])
       data = prop(wavelet.to(device), self.s_cor.to(device), self.r_cor.to(device), self.dt).cpu()
       return data
    
    


    def run_inversion(self,model,data_t,wavelet,msk,niter,device=None,**kwargs): 
       """ 
      This run the FWI inversion,  
      ===================================
      Arguments: 
         model: torch.Tensor [nz.nx]: 
            Initial model for FWI 
         data_t: torch.Tensor [nt,ns,nr]: 
            Observed data
         wavelet: torch.Tensor [nt,1,1] or [nt,ns,1]
            wavelet 
         msk: torch.Tensor [nz,nx]:
            Mask for water layer
         niter: int: 
            Number of iteration 
         device: gpu or cpu  
       ==================================
      Optional: 
         vmin: int:
            upper bound for the update 
         vmax: int: 
            lower bound for the update 
         smth_flag: bool: 
            smoothin the gradient flag 
         smth: sequence of tuble or list: 
            each element define the amount of smoothing for different axes
         hvd: bool: 
            Use horovod for multi-GPU implimintation 
         tv_flag: bool:
             Flag for adding TV reg
         alphatv:  float:
             Tv coefficient 
         plot_flag: bool 
       """

       # Defining parameters 
       model = model.to(device)
       wavelet = wavelet.to(device)
       msk = torch.from_numpy(msk).int().to(device)
       model.requires_grad=True 
       m_max = kwargs.pop('vmax', 4.5)
       m_min = kwargs.pop('vmin', 1.5)
       smth_flag = kwargs.pop('smth_flag', False)
       hvd_flag = kwargs.pop('hvd_flag', False)
       tv_flag = kwargs.pop('tv_flag', False)
       plot_flag = kwargs.pop('plot_flag', False)
       if tv_flag: 
           alphatv = kwargs.pop('alphatv', None)
           assert alphatv != None, " specify TV coefficient 'alphatv' "
       if smth_flag: 
          smth = kwargs.pop('smth', '')
          assert smth != '', " 'smth' is not specified "
        
        
        
       # Defining objective and step-length  
       criterion = torch.nn.MSELoss()
       LR = 0.01
#        LR = 0.1
       optimizer = torch.optim.Adam([{'params':[model],'lr':LR}])
       scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=20,threshold=1e-3,verbose=False)		
#        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,niter, verbose=True)

       num_batches = self.num_batches
       num_shots_per_batch = int(self.num_shots / num_batches)
       prop = deepwave.scalar.Propagator({'vp': model}, self.dx,pml_width=[0,20,20,20,20,20])
       t_start = time.time()
       loss_iter=[]
       increase = 0
       min_loss = 0
       first = True
       tol = 1e-4
   
       # updates is the output file
       updates=[]
       relax = 0
        # main inversion loop 
        
       if hvd_flag: 
           for itr in range(niter):
              running_loss = 0 
              optimizer.zero_grad()

              for ibatch,(batch_data_t,batch_x_s,batch_x_r) in enumerate(data_t): 
                 shot_indices =  ( torch.round ( (batch_x_s[:, 0, 1]  - self.os  + self.ox )/ self.ds )).int().numpy()
                 if shot_indices.shape[0] > 1:  
                    batch_wavl = wavelet[:, shot_indices, 0].view(wavelet.shape[0],shot_indices.shape[0],1)
                 else: 
                       batch_wavl = wavelet[:, shot_indices, 0].view(wavelet.shape[0],1,1)  
                 batch_data_t = batch_data_t.cuda()
                 batch_data_t = batch_data_t.permute(1,0,2)
                 batch_x_s    = batch_x_s.cuda()
                 batch_x_r    = batch_x_r.cuda()
                 batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt)
                 loss = criterion(batch_data_pred, batch_data_t) 
                 loss.backward()
                 running_loss += loss.item()  
              running_loss = self.metric_average(running_loss, 'avg_loss') 
              model.grad = hvd.allreduce(model.grad.contiguous())      
              if smth_flag: model.grad = self.grad_smooth(model,smth[1],smth[0]).to(device)            
              if itr == 0 : gmax0 = (torch.abs(model.grad)).max() # get max of first itr 
#           model.grad = model.grad / gmax0   # normalize                 
              model.grad =  self.grad_reg(model,mask=msk)        
              if tv_flag: 
                   mTV = model.detach().clone()
                   mTV.requires_grad= True 
                   lossTV = tv_loss(mTV)
                   lossTV.backward()                     
#                 gmax   = (torch.abs(model.grad)).max()  # to make the TVgrad in the same order of that of the data 
#                 gTVmax  = torch.abs(mTV.grad).max()  # get max of TV 
#                 mTV.grad = mTV.grad/gTVmax * gmax  # Normalize TV
                   mTV.grad = self.grad_reg(mTV,msk)
                   if itr>0 and itr%50==0: alphatv = alphatv/2
                   model.grad = (model.grad + alphatv * mTV.grad)   # combine the gradient of the two 
              model.grad =  self.grad_reg(model,mask=msk)
#           model.grad = model.grad * msk  # mask the grad                
              optimizer.step()   
              scheduler.step(running_loss)
              model.data[model.data < m_min] = m_min
              model.data[model.data > m_max] = m_max
              loss_iter.append(running_loss)
              if hvd.local_rank()==0: 
                    print('Iteration: ', itr, 'Objective: ', running_loss)
                    if itr%5==0: 
                        updates.append(model.detach().clone().cpu().numpy())       
                    
              if plot_flag and itr%5==0 and hvd.local_rank()==0:
                   plt.figure(figsize=(10,5))
                   plt.imshow(model.grad.cpu().numpy(),cmap='seismic',vmin=-0.3,vmax=0.3)
                   plt.colorbar(shrink=0.3)
                   plt.show()
    #                plt.close()
                   plt.figure(figsize=(10,5))
                   plt.imshow(model.detach().clone().cpu().numpy(),cmap='jet',vmax=4.5,vmin=1.5)
                   plt.colorbar(shrink=0.3)
                   plt.show()
    #                plt.close()
                   
              # stopping criteria or relax condition smoothing ()
              if np.abs(loss_iter[itr] - loss_iter[itr-1])/max(loss_iter[itr],loss_iter[itr-1]) < tol and itr>20: 
                  if smth_flag and relax < 6:
                    smth[0]=smth[0]/2
                    smth[1]=smth[1]/2
                    relax +=1
                  elif  tv_flag and relax <6:
                    alphatv = alphatv/2
                    relax +=1
                  else:                
                      t_end = time.time()
                      if hvd.local_rank()==0:
                         print('Runtime in min :',(t_end-t_start)/60)   
                      updates.append(model.detach().clone().cpu().numpy()) 
                      return np.array(updates),loss_iter
                    
              # early stopping       
              elif min_loss < loss_iter[itr] and itr > 20: 
                 increase +=1
              else: 
                 increase = 0
                 min_loss = loss_iter[itr] 
              if  increase == 10: 
                  if smth_flag and relax < 6:
                    smth[0]=smth[0]/2
                    smth[1]=smth[1]/2
                    relax +=1
                    increase =5                
                  elif  tv_flag and relax <6:
                    alphatv = alphatv/2
                    relax +=1
                    increase=5
                  else:
                      t_end = time.time()
                      if hvd.local_rank()==0:
                         print('Runtime in min :',(t_end-t_start)/60)  
                      updates.append(model.detach().clone().cpu().numpy()) 
                      return np.array(updates),loss_iter
                        
       # If not hvd
       else: 
           for itr in range(niter):
              running_loss = 0 
              optimizer.zero_grad() 

              for it in range(num_batches): # loop over shots 
                 # batch_wavl = wavelet.repeat(1, num_shots_per_batch, 1)
                 batch_wavl = wavelet[:,it::num_batches]
                 batch_data_t = data_t[:,it::num_batches].to(device)
                 batch_x_s = self.s_cor[it::num_batches].to(device)
                 batch_x_r = self.r_cor[it::num_batches].to(device)
                 batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt)        
                 loss = criterion(batch_data_pred, batch_data_t)
                 if loss.item() == 0.0: 
                    updates.append(model.detach().cpu().numpy())
                    return np.array(updates)
                 loss.backward()            
                 running_loss += loss.item()             
                             

              if smth_flag: model.grad = self.grad_smooth(model,smth[1],smth[0]).to(device)            
              if itr == 0 : gmax0 = (torch.abs(model.grad)).max() # get max of first itr 
#           model.grad = model.grad / gmax0   # normalize                 
              model.grad =  self.grad_reg(model,mask=msk)
        
              if tv_flag: 
                   mTV = model.detach().clone()
                   mTV.requires_grad= True 
                   lossTV = tv_loss(mTV)
                   lossTV.backward()                                  
#                 gmax   = (torch.abs(model.grad)).max()  # to make the TVgrad in the same order of that of the data 
#                 gTVmax  = torch.abs(mTV.grad).max()  # get max of TV 
#                 mTV.grad = mTV.grad/gTVmax * gmax  # Normalize TV
                   mTV.grad = self.grad_reg(mTV,msk)
#                    if itr>0 and itr%50==0: alphatv = alphatv/2
                   model.grad = (model.grad + alphatv * mTV.grad)   # combine the gradient of the two 
              model.grad =  self.grad_reg(model,mask=msk)
#           model.grad = model.grad * msk  # mask the grad
               
              optimizer.step()   
              scheduler.step(running_loss)
              model.data[model.data < m_min] = m_min
              model.data[model.data > m_max] = m_max
              loss_iter.append(running_loss)
              print('Iteration: ', itr, 'Objective: ', running_loss) 

                
              if plot_flag and itr%5==0:
                   plt.figure(figsize=(10,3))
                   plt.imshow(model.grad.cpu().numpy(),cmap='seismic',vmin=-0.3,vmax=0.3)
                   plt.axis('tight')
                   plt.colorbar(shrink=0.2)
                   plt.show()
    #                plt.close()
                   plt.figure(figsize=(10,3))
                   plt.imshow(model.detach().clone().cpu().numpy(),cmap='jet',vmax=4.5,vmin=1.5)
                   plt.axis('tight')
                   plt.colorbar(shrink=0.2)
                   plt.show()
    #                plt.close()
            
            
              if itr > 0 and itr%1==0 :
                    updates.append(model.detach().clone().cpu().numpy())  
                    
              # stopping criteria or relax condition smoothing ()
              if np.abs(loss_iter[itr] - loss_iter[itr-1])/max(loss_iter[itr],loss_iter[itr-1]) < tol and itr>20: 
#                   if smth_flag or tv_flag: 
                  if relax < 6:
                        if smth_flag: 
                            smth[0]=smth[0]/2
                            smth[1]=smth[1]/2
                            increase =5
                        if tv_flag :
                            alphatv = alphatv/2
                            increase =5
                        relax += 1     
                        print(f"Reduce the smoothing from {smth[0]*2},{smth[1]*2} to {smth[0]},{smth[1]}")
                  else:
                      t_end = time.time()
                      print('Runtime in min :',(t_end-t_start)/60)  
                      updates.append(model.detach().clone().cpu().numpy()) 
                      return np.array(updates),loss_iter
              # early stopping       
              elif min_loss < loss_iter[itr] and itr > 20: 
                 increase +=1
              else: 
                 increase = 0
                 min_loss = loss_iter[itr] 
              if  increase == 10: 
#                   if smth_flag or tv_flag: relax +=1 
                  if relax < 5:
                        if smth_flag: 
                            smth[0]=smth[0]/2
                            smth[1]=smth[1]/2
                            increase =5
                        if tv_flag :
                            alphatv = alphatv/2
                            increase =5
                        relax += 1     
                        print(f"Reduce the smoothing from {smth[0]*2},{smth[1]*2} to {smth[0]},{smth[1]}")
                  else:
                      t_end = time.time()
                      print('Runtime in min :',(t_end-t_start)/60)  
                      updates.append(model.detach().clone().cpu().numpy()) 
                      return np.array(updates),loss_iter
       # End of FWI iteration
       t_end = time.time()
       if hvd_flag==True and hvd.local_rank()==0:
            print('Runtime in min :',(t_end-t_start)/60)
       elif hvd_flag==False:
           print('Runtime in min :',(t_end-t_start)/60)         
       updates.append(model.detach().clone().cpu().numpy()) 
       return np.array(updates),loss_iter

    
    def grad_smooth(self,model,sigmax,sigmaz):
               m =  model.detach().clone().cpu()
               gradient = model.grad.cpu().numpy() 
               gradient = gaussian_filter1d(gradient,sigma=sigmaz,axis=0) # z
               gradient = gaussian_filter1d(gradient,sigma=sigmax,axis=1) # x 
               gradient = torch.tensor(gradient)
               return gradient


    def grad_reg(self,model,mask):
               # m =  model.detach().clone().cpu()
                       
               gradient = model.grad
               
               gmax     = (torch.abs(gradient)).max() 
               gradient = gradient / gmax  # normalize the gradient 
               
               gradient = gradient * mask

               return gradient
            
    def metric_average(self,val, name):
               tensor = torch.tensor(val)
               avg_tensor = hvd.allreduce(tensor, name=name)
               return avg_tensor.item()



def tv_loss(model):
        h, w = model.shape 
        a = torch.square(model[:h - 1, :w - 1] - model[1:, :w - 1])
        b = torch.square(model[:h - 1, :w - 1] - model[:h - 1, 1:])
        return torch.sum(torch.pow(b +   a + 1e-15, 0.6))/(h*w)
      