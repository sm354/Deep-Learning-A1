class LayerNorm_LSTMCell(nn.Module):
  def __init__(self, embedding_size, hidden_size, epsilon=1e-5):
    super(LayerNorm_LSTMCell, self).__init__()
    self.embedding_size=embedding_size
    self.hidden_size=hidden_size
    self.epsilon=1e-5

    self.inp2hidden = nn.Linear(embedding_size, 4*hidden_size) #since there are 4 linear transformations : i,f,g,o
    self.hidden2hidden = nn.Linear(hidden_size, 4*hidden_size) #this takes input h_t_minus_1

    # define layernorm layers : we are normalising input to input,forget,output,cell gate in "cummulative fashion"
    self.layernorm_inp2hidden = nn.LayerNorm(4*hidden_size)
    self.layernorm_hidden2hidden = nn.LayerNorm(4*hidden_size)
    self.layernorm_over_c = nn.LayerNorm(hidden_size)
   
  def forward(self, inp, h_c_prev=None): # x (batch,embedding_size), h_c (h and c at t-1) : [(batch,hidden_size)]*2
    h_prev, c_prev = h_c_prev if h_c_prev!=None else (torch.zeros((inp.shape[0],self.hidden_size)).to(device) , torch.zeros((inp.shape[0],self.hidden_size)).to(device))
    # first do the linear transformations with layernorms
    linear_transform_inp = self.layernorm_inp2hidden(self.inp2hidden(inp)) + self.layernorm_hidden2hidden(self.hidden2hidden(h_prev))

    #find the gates
    split=self.hidden_size
    i=torch.sigmoid(linear_transform_inp[:,0:split])
    f=torch.sigmoid(linear_transform_inp[:,split:2*split])
    o=torch.sigmoid(linear_transform_inp[:,2*split:3*split])
    g=torch.tanh(linear_transform_inp[:,3*split:])
    c=f*c_prev+i*g

    #apply final layer norm
    c=self.layernorm_over_c(c)
    h=o*torch.tanh(c)
    return (h,c)

class LSTM(nn.Module):
  def __init__(self, embedding_size, hidden_size, bidirectional=True, batch_first=True):
    super(LSTM, self).__init__()
    # bidirectional=True, batch_first=True : always (assumed)
    self.embedding_size=embedding_size
    self.hidden_size=hidden_size
    self.lstm_cell_fwd=LayerNorm_LSTMCell(embedding_size, hidden_size)
    self.lstm_cell_bkwd=LayerNorm_LSTMCell(embedding_size, hidden_size)

  def forward(self, inp): #inp (batch,seq_len,embedding_size)
    # iterate over seq_len in forward and backward direction (using their respective lstm cells)

    ht_fwd, ht_bkwd = [],[]
    ct_fwd, ct_bkwd = [],[]
    
    for t in range(inp.shape[1]):
      #forward lstm
      h_c_prev=(ht_fwd[t-1],ct_fwd[t-1]) if t!=0 else None
      ht,ct=self.lstm_cell_fwd(inp[:,t,:],h_c_prev)
      ht_fwd.append(ht)
      ct_fwd.append(ct)

      #backward lstm
      t_ = inp.shape[1]-t-1
      h_c_prev_ = (ht_bkwd[0],ct_bkwd[0]) if t_!=inp.shape[1]-1 else None
      ht_,ct_=self.lstm_cell_bkwd(inp[:,t_,:],h_c_prev_)
      ht_bkwd.insert(0,ht_)
      ct_bkwd.insert(0,ct_)
    
    #now stack ht and give it as out
    #also take last ht,ct of fwd lstm and first ht,ct of bkwd lstm
    #and give it as hn,cn after stacking

    ht_fwd=torch.stack(ht_fwd) #seq_len,batch,hidden_dim
    ct_fwd=torch.stack(ct_fwd) #seq_len,batch,hidden_dim
    ht_bkwd=torch.stack(ht_bkwd) #seq_len,batch,hidden_dim
    ct_bkwd=torch.stack(ct_bkwd) #seq_len,batch,hidden_dim
    out=torch.cat((ht_fwd,ht_bkwd),dim=2).permute(1,0,2) #batch,seq_len,2*hidden_dim
    
    hn=torch.stack((ht_fwd[-1],ht_bkwd[0])) #batch,hidden_dim
    cn=torch.stack((ct_fwd[-1],ct_bkwd[0])) #batch,hidden_dim
    return out,(hn,cn)