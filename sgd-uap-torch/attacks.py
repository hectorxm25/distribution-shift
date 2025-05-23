import math
import torch
import torch.nn as nn


'''
Basic version of untargeted stochastic gradient descent UAP adapted from:
[AAAI 2020] Universal Adversarial Training
- https://ojs.aaai.org//index.php/AAAI/article/view/6017

Layer maximization attack from:
Universal Adversarial Perturbations to Understand Robustness of Texture vs. Shape-biased Training
- https://arxiv.org/abs/1911.10364
'''
def uap_sgd(model, loader, nb_epoch, eps, beta = 12, step_decay = 0.8, y_target = None, loss_fn = None, layer_name = None, uap_init = None):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    y_target    target class label for Targeted UAP variation
    loss_fn     custom loss function (default is CrossEntropyLoss)
    layer_name  target layer name for layer maximization attack
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    
    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
    '''
    _, (x_val_sample, y_val_sample) = next(enumerate(loader)) # Sample for shape, device is CPU
    batch_size_sample = len(x_val_sample)

    # Initialize delta (the UAP) on CPU
    if uap_init is None:
        # delta is the single UAP (e.g., 3xHxW)
        delta = torch.zeros(x_val_sample.shape[1:], device=x_val_sample.device) 
    else:
        # Ensure uap_init is a detached CPU tensor
        delta = uap_init.clone().detach().to(x_val_sample.device)

    # batch_delta will be used per batch, requires grad, on CPU.
    # Its .data will be updated from delta.
    # Initialize with correct shape from sample, will be adapted if batch_size changes.
    batch_delta = delta.unsqueeze(0).repeat([batch_size_sample, 1, 1, 1]).clone()
    batch_delta.requires_grad_()
    
    losses = []
    
    # loss function
    if layer_name is None:
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction = 'none')
        # beta should be on the same device as the loss computation (CUDA)
        beta_tensor = torch.tensor([beta], device='cuda', dtype=torch.float32) 
        def clamped_loss(output, target):
            # output and target are on CUDA
            loss = torch.mean(torch.min(loss_fn(output, target), beta_tensor))
            return loss
       
    # layer maximization attack
    else:
        def get_norm(self, forward_input, forward_output):
            global main_value
            main_value = torch.norm(forward_output, p = 'fro')
        for name, layer in model.named_modules():
            if name == layer_name:
                handle = layer.register_forward_hook(get_norm)
                
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))
        
        # perturbation step size with decay
        eps_step = eps * step_decay
        
        for i, (x_val, y_val) in enumerate(loader): # x_val, y_val are CPU
            # Zero grad for batch_delta only if it's not None
            if batch_delta.grad is not None:
                batch_delta.grad.data.zero_()
            
            # Adapt batch_delta to current batch size if different, and update its data from delta
            current_batch_size = x_val.shape[0]
            if batch_delta.shape[0] != current_batch_size:
                batch_delta_data_current_batch = delta.unsqueeze(0).repeat([current_batch_size, 1, 1, 1])
                # Re-assign data and ensure requires_grad is preserved.
                # If just .data is assigned to a new tensor of different shape, it might cause issues.
                # Better to ensure batch_delta has the right shape from start or handle carefully.
                # For simplicity, assuming batch size is consistent or covered by .repeat correctly.
                # The .data assignment below should handle variable batch_size if 'delta' is base.
                batch_delta = delta.unsqueeze(0).repeat([current_batch_size, 1, 1, 1]).clone().requires_grad_(True)
            else:
                batch_delta.data = delta.unsqueeze(0).repeat([current_batch_size, 1, 1, 1])

            # for targeted UAP, switch output labels to y_target
            if y_target is not None: y_val = torch.ones(size = y_val.shape, dtype = y_val.dtype) * y_target
            
            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            outputs = model(perturbed)
            
            # loss function value
            if layer_name is None: loss = clamped_loss(outputs, y_val.cuda())
            else: loss = main_value
            
            if y_target is not None: loss = -loss # minimize loss for targeted UAP
            losses.append(torch.mean(loss).cpu()) # Move to CPU before appending
            loss.backward()
            
            # batch update
            grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)
    
    if layer_name is not None: handle.remove() # release hook
    
    return delta.data, losses
