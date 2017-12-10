def cnnModel1(dataType,trainSize,input_placeholder,activation,init,targets,fftSize,padding,keep_prob1, keep_prob2, keep_prob3):
    
    if dataType=='test':
        targets=2    
    
    f=512  # lets take 512 as default
    
    # Input = 100x257
    if fftSize == 512:
        f = 257        
    elif fftSize == 256:
        f = 129
    elif fftSize == 1024:
        f = 513
    elif fftSize == 2048:
        f = 1025   
    
    weight_list = list()
    activation_list = list()
    bias_list = list()
    
    if trainSize == '1sec':
        time_dim=100
        if activation=='mfm':
            fc_input=16448 #1*257*64 = 16448
        else:
            fc_input= 32896 # 1*257*128

    elif trainSize == '2sec':
        time_dim=200
        if activation=='mfm':
            fc_input=0  #Check it   
        else:
            fc_input=0 # chek it
            
    elif trainSize == '3sec':
        time_dim=300
        if activation=='mfm':
            fc_input=5280   #10*33*16
        else:
            fc_input=10560  #10*33*32            
            
    elif trainSize == '4sec':
        time_dim = 400
        if activation == 'mfm':
            fc_input = 6864    #13*33*16 = 6864
        else:
            fc_input = 13728   #13*33*32 = 13728
        
    elif trainSize == '5sec':
        time_dim = 500
        if activation == 'mfm':
            fc_input = 8976    #17*33*16
        else:
            fc_input = 17952   #17*33*32
                
    if activation=='mfm':
        in_conv2 = 64
        in_conv3 = 64
        in_conv4 = 64
        in_fc2 = 128
        in_fc3 = 128
        in_outputLayer = 128
    else:
        in_conv2 = 128
        in_conv3 = 128
        in_conv4 = 128
        in_fc2 = 256
        in_fc3 = 256
        in_outputLayer = 256
                       
    #Convolution layer1,2,3    
    conv1,w1,b1 = conv_layer(input_placeholder, [3,f,1,128], [128], [1,1,1,1],'conv1',padding,activation,init)
    weight_list.append(w1)
    bias_list.append(b1)
    print('Conv1 ', conv1)
    
    conv2,w2,b2 = conv_layer(conv1, [3,1,in_conv2,128], [128], [1,1,1,1],'conv2', padding,activation,init)
    weight_list.append(w2)
    bias_list.append(b2)    
    print('Conv2 ', conv2)
    
    conv3,w3,b3 = conv_layer(conv2, [3,1,in_conv3,128], [128], [1,1,1,1],'conv3', padding,activation,init)
    weight_list.append(w3)
    bias_list.append(b3)    
    print('Conv2 ', conv3)
    
    #Max-pooling layer over time    
    pool1 = maxPool2x2(conv3, [1,time_dim,1,1], [1,time_dim,1,1])
    print('Pool1 layer shape = ', pool1)
    
    #100x257x64 is input to maxpool
    #output = 1X257x64
    # 1*257*64 = 16448

    # Dropout on the huge input from Conv layer    
    flattened = tf.reshape(pool1, shape=[-1,fc_input])
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')
    
    # Fully connected layer 1 with 256 neurons but gets splitted into 128 due to MFM
    fc1,w4,b4, = fc_layer(dropped_1, fc_input, 256, 'FC_Layer1', activation)
    weight_list.append(w4)
    bias_list.append(b4)
    
    print('Shape of FC1 = ', fc1.shape)
    
    # Dropout followed by FC layer with 256 neurons but gets splitted into 128 due to MFM
    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')        
    fc2,w5,b5, = fc_layer(dropped_2, in_fc2, 256, 'FC_Layer2', activation)
    weight_list.append(w5)
    bias_list.append(b5)
    
    print('Shape of FC2 = ', fc2.shape)

    # Dropout followed by FC layer with 256 neurons but gets splitted into 128 due to MFM
    dropped_3 = drop_layer(fc2, keep_prob2, 'dropout3')        
    fc3,w6,b6, = fc_layer(dropped_3, in_fc3, 256, 'FC_Layer3', activation)
    weight_list.append(w6)
    bias_list.append(b6)
    
    print('Shape of FC3 = ', fc3.shape)

    #Output layer: 2 neurons. One for genuine and one for spoof. Dropout applied first
    dropped_4 = drop_layer(fc3, keep_prob3, 'dropout4')
    output=None
    w7=None
    b7=None
    
    if targets == 2:
        output,w7,b7 = fc_layer(dropped_4, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits
    elif targets == 4:        
        output,w7,b7 = fc_layer(dropped_4, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits
    elif targets == 11:
        output,w7,b7 = fc_layer(dropped_4, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits
    elif targets == 14:
        output,w7,b7 = fc_layer(dropped_4, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits
        
    weight_list.append(w7)
    bias_list.append(b7)            
                
    print('Output layer shape = ', output.shape)
    
    
    return fc3, output, weight_list, activation_list, bias_list