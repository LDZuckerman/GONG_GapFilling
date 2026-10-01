import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

############
# Simple CNN
############

class SimpleCNN(nn.Module):
    '''
    *Very* simple CNN for [N_img, N_pix, N_pix] -> [N_img, N_pix, N_pix] 
    '''
    
    def __init__(self, len_set, k_size=3, padding_mode='zeros', hidden_channels=[16, 64]):
        super(SimpleNet, self).__init__()
        
        self.act = nn.ReLU()
        self.convs = nn.ModuleList()
        
        in_chans = len_set
        for hc in hidden_channels:
            self.convs.append(nn.Conv2d(in_channels=in_chans, out_channels=hc, kernel_size=k_size, stride=1, padding=1, padding_mode=padding_mode))
            self.convs.append(self.act)
            in_chans = hc
            
        self.convs.append(nn.Conv2d(in_channels=hidden_channels[-1], out_channels=len_set, kernel_size=k_size, stride=1, padding=1, padding_mode=padding_mode))
        self.convs.append(self.act)
        
    def forward(self, x):

        # Copy input
        x_in = x.clone().float()

        # Perform convolutions
        for conv in self.convs:
            x = conv(x.float())

        ##Could try: Force output to equal input where no missing data - 
        #already_filled_idxs = torch.sum(x_in, dim=(0,2,3)) != 0
        #x_out = x.clone()
        #x_out[:,already_filled_idxs,:,:] = x_in[:,already_filled_idxs.numpy(),:,:]
        x_out = x

        return x_out
    

    
############
# Simple RNN 
# NOTE: RNN layers require 3D input, so first need to use conv lauyers to get rid of channel dim
############

class SimpleRNN(nn.Module):
    '''
    RNN for for [N_img, N_pix, N_pix] -> [N_img, N_pix, N_pix] 
    '''

    def __init__(self, n_pix, hidden_channels=[16, 32], num_layers=2):
        super(SimpleRNN, self).__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, hidden_channels[0], kernel_size=3, stride=2, padding=1),  # [16*15, 16, 64, 64]
                                 nn.ReLU(),
                                 nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=2, padding=1), # [16*15, 32, 32, 32]
                                 nn.ReLU())     

        self.rnn = nn.RNN(input_size=hidden_channels[-1]**3, hidden_size=hidden_channels[-1]**3, batch_first=True) # GETS KILLED WHEN I USED NUM_LAYERS KEYWORD? 
 
        self.dec = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1))
        

    def forward(self, x):
        
        x = x.float()
        
        # Reshape
        n_batch, n_series, n_pix, _ = x.shape
        x = x.view(n_batch * n_series, 1, n_pix, n_pix) # [240, 1, 128, 128]
        
        # Perform encode convs
        x = self.enc(x.float()) # [240, 32, 32, 32]
        
        # Reshape 
        x = x.view(n_batch * n_series, -1) # [240, 32768] 
        x = x.view(n_batch, n_series, -1) # [16, 15, 32768]
        
        # Perforn RNN
        x, _ = self.rnn(x) # [16, 15, 32768] 
        
        # Reshape
        x = x.reshape(n_batch * n_series, 32, 32, 32) # [240, 32, 32, 32]

        # Perform decode convs
        x = self.dec(x)
        
        # Reshape
        x = x.view(n_batch, n_series, n_pix, n_pix)
        

        return x
    
    
#######
# UNet
######   

# class UNet(nn.Module):
#     '''
#     UNet for [N_img, N_pix, N_pix] -> [N_img, N_pix, N_pix] 
#     '''
    
#     def __init__(self, len_set, hidden_channels=[64, 128, 256, 512],): # len_set, k_size=3, padding_mode='zeros', hidden_channels=[16, 64])
        
#         super(UNet, self).__init__() 
        
#         self.ups = nn.ModuleList() 
#         self.downs = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
#         in_channels = len_set
#         for n_chans in hidden_channels:
#             self.downs.append(DoubleConv(in_channels, n_chans))
#             in_channels = n_chans  # after each convolution we set (next) in_channel to (previous) out_channels 
            
#         for n_chans in reversed(hidden_channels):
#             self.ups.append(nn.ConvTranspose2d(n_chans*2, n_chans, kernel_size=2, stride=2,))
#             self.ups.append(DoubleConv(n_chans*2, n_chans))
            
#         self.bottleneck = DoubleConv(hidden_channels[-1], hidden_channels[-1]*2)
#         self.final_conv = nn.Conv2d(hidden_channels[0], len_set, kernel_size=1)
    

#     def forward(self, x): 
        
#         x = x.to(torch.float32)

#         # Perform downs
#         skip_connections = []
#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x) 
#             x = self.pool(x)
#         x = self.bottleneck(x)
        
#         # Reverse skip connections
#         skip_connections = skip_connections[::-1] # reverse 
        
#         # Perform ups
#         for idx in range(0, len(self.ups), 2): # step of 2 becasue add conv step
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx//2]
#             if x.shape != skip_connection.shape:
#                 x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)
#             concat_skip = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx+1](concat_skip)

#         return self.final_conv(x)


class UNet_n(nn.Module):
    '''
    UNet for [N_img, N_pix, N_pix] -> [N_img, N_pix, N_pix] 
    '''
    
    def __init__(self, len_set, hidden_channels=[64, 128, 256, 512], convblock_depth=2, only_centers=False, only_outers=False): # len_set, k_size=3, padding_mode='zeros', hidden_channels=[16, 64])
        
        super(UNet_n, self).__init__() 
        
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.only_centers = only_centers
        self.only_outers = only_outers
        
        in_channels = len_set
        for n_chans in hidden_channels:
            if convblock_depth == 2: # for backwards compatability
                self.downs.append(DoubleConv(in_channels, n_chans))
            else:
                self.downs.append(MultiConv(in_channels, n_chans, convblock_depth))
            in_channels = n_chans  # after each convolution we set (next) in_channel to (previous) out_channels 
            
        for n_chans in reversed(hidden_channels):
            self.ups.append(nn.ConvTranspose2d(n_chans*2, n_chans, kernel_size=2, stride=2,))
            if convblock_depth == 2: # for backwards compatabilty
                self.ups.append(DoubleConv(n_chans*2, n_chans))
            else:
                self.ups.append(MultiConv(n_chans*2, n_chans, convblock_depth))
            
        self.bottleneck = DoubleConv(hidden_channels[-1], hidden_channels[-1]*2)
        self.final_conv = nn.Conv2d(hidden_channels[0], len_set, kernel_size=1)
    

    def forward(self, x): 
        
        x = x.to(torch.float32)
        #print('input', x.shape)

        # Perform downs
        skip_connections = []
        for down in self.downs:
            x = down(x)
            #print('after down', x.shape)
            skip_connections.append(x) 
            x = self.pool(x)
            #print('after pool', x.shape)
        x = self.bottleneck(x)
        #print('after bottleneck', x.shape)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1] # reverse 
        
        # Perform ups
        for idx in range(0, len(self.ups), 2): # step of 2 becasue add conv step
            x = self.ups[idx](x)
            #print('after up', x.shape)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            #print('after skip add', concat_skip.shape)
            x = self.ups[idx+1](concat_skip)
            #print('after up', x.shape)
            
        # If desired, force outer or inner regions to be zeros
        if self.only_centers or self.only_centers:
            n_pix = x.shape[2]
            xx, yy = np.meshgrid(np.linspace(0, n_pix-1, n_pix), np.linspace(0, n_pix-1, n_pix), indexing='ij')
            ctr_x, ctr_y = n_pix/2, n_pix/2
            r = np.sqrt(((xx-ctr_x)**2 + (yy-ctr_y)**2))
            mask = torch.zeros_like(x, dtype=torch.bool) 
            if self.only_centers:
                use_idxs = np.where(r < 33)
            if self.only_outers:
                use_idxs = np.where(r > 27)               
            mask[:, :, torch.from_numpy(use_idxs[0]), torch.from_numpy(use_idxs[1])] = True
            x = torch.where(mask, x, 0)

        return self.final_conv(x)
    

class DoubleConv(nn.Module):
    '''
    Containor for conv sets (for convenience)
    '''
    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)
    
    
class MultiConv(nn.Module):
    '''
    Containor for conv sets (for convenience)
    '''
    def __init__(self, in_channels, out_channels, convblock_depth):

        super(MultiConv, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False))
        self.convs.append(nn.BatchNorm2d(out_channels))
        self.convs.append(nn.ReLU(inplace=True))
        for i in range(convblock_depth-1):
            self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False))
            self.convs.append(nn.BatchNorm2d(out_channels))
            self.convs.append(nn.ReLU(inplace=True))

    def forward(self, x):
        
        for conv in self.convs:
            x = conv(x)
        
        return x


    
###########
# BRITS?
#   - Below is for reference only, repoduced from https://github.com/NIPS-BRITS/BRITS/blob/master/models/brits.py
#   - Original is for [N_batch, N_obs_per_TS, N_feat_per_obs] --> [] (I think)
#   - Need to modify to do [N_batch, N_img_per_TS, N_pix, N_pix] -> [N_batch, N_img_per_TS, N_pix, N_pix]
###########   
    
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.rits_f = rits_model()
        self.rits_b = rits_model()

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
    
    
class rits_model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(35 * 2, RNN_HID_SIZE)

        self.temp_decay_h = TemporalDecay(input_size = 35, output_size = RNN_HID_SIZE, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = 35, output_size = 35, diag = True)

        self.hist_reg = nn.Linear(RNN_HID_SIZE, 35)
        self.feat_reg = FeatureRegression(35)

        self.weight_combine = nn.Linear(35 * 2, 35)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(RNN_HID_SIZE, 1)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = F.sigmoid(y_h)

        return {'loss': x_loss / SEQ_LEN + y_loss * 0.3, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
############ ############# SUGGESTED ############ ############

class CNNRNN_Autoencoder(nn.Module):
    def __init__(self, rnn_hidden=64):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [N, 1, 128, 128] → [N, 16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [N, 16, 64, 64] → [N, 32, 32, 32]
            nn.ReLU()
        )

        self.enc_h = 32
        self.enc_w = 32
        self.enc_c = 32
        self.feature_dim = self.enc_c * self.enc_h * self.enc_w  # 32*32*32 = 32768

        # -------- RNN --------
        self.rnn = nn.RNN(
            input_size=self.feature_dim,  # 32768
            hidden_size=self.feature_dim, # 32768
            batch_first=True
        )

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # [N, 32, 32, 32] → [N, 16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # [N, 16, 64, 64] → [N, 1, 128, 128]
        )

    def forward(self, x):
        # x shape: [B, T, H, W]
        # x: [16, 15, 128, 128]

        B, T, H, W = x.shape

        # --------------------------------------------------
        # Encode (spatial)
        # --------------------------------------------------

        # Merge batch and time, add channel dimension
        x = x.view(B * T, 1, H, W)
        # x: [16*15, 1, 128, 128] = [240, 1, 128, 128]

        # Apply CNN encoder
        x = self.encoder(x)
        # x: [240, 32, 32, 32]

        # Flatten spatial dimensions
        x = x.view(B * T, -1)
        # x: [240, 32768]

        # --------------------------------------------------
        # Temporal modeling (RNN)
        # --------------------------------------------------

        # Restore batch and time dimensions
        x = x.view(B, T, -1)
        # x: [16, 15, 32768]

        # Apply RNN across time
        x, _ = self.rnn(x)
        # x: [16, 15, 32768]

        # --------------------------------------------------
        # Decode (spatial reconstruction)
        # --------------------------------------------------

        # Merge batch and time again, restore feature maps
        x = x.view(B * T, self.enc_c, self.enc_h, self.enc_w)
        # x: [240, 32, 32, 32]

        # Apply CNN decoder
        x = self.decoder(x)
        # x: [240, 1, 128, 128]

        # Restore original batch and time structure
        x = x.view(B, T, H, W)
        # x: [16, 15, 128, 128]

        return x

    
    
 