# !python3 -m pip install --upgrade pip
# !pip3 install torch
# !pip install transformers
# !pip install tensorflow-gpu
# !pip install sentencepiece
# !pip3 install matplotlib
# !pip3 install transformers==2.9.0 
# !pip3 install pytorch_lightning==0.7.5

import sentencepiece

import pandas as pd
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import  Adafactor 
import time
import warnings
import re
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

import torch
if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")





# ------------------------------------------------------------    
# ------------------------------------------------------------    
from IPython.display import HTML, display
def progress(loss,value, max=100):
    return HTML(""" Batch loss :{loss}
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(loss=loss,value=value, max=max))


def train_model(inputdata, pretrain,file_name,key_in, key_out, n_epochs):
    tokenizer = T5Tokenizer.from_pretrained('t5-small', local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(pretrain, local_files_only=True)#, return_dict=True)
    model.to(dev)
    
    model.train()
    optimizer = Adafactor(
    model.parameters(),
    lr=1e-4,#1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=0.9, #None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)
    num_of_epochs=n_epochs
    train_df = pd.read_csv(inputdata)
    train_df = train_df.fillna('')
    batch_size=2
    num_of_batches=len(train_df)/batch_size
    num_of_epochs=1
    num_of_batches=int(num_of_batches)
    print(batch_size, num_of_batches, num_of_epochs)



    loss_per_10_steps=[]
    for epoch in range(1,num_of_epochs+1):
        print('Running epoch: {}'.format(epoch))

        running_loss=0

        # out = display(progress(1, num_of_batches+1), display_id=True)
        for i in range(num_of_batches):
            inputbatch=[]
            labelbatch=[]
            new_df=train_df[i*batch_size:i*batch_size+batch_size]
            for indx,row in new_df.iterrows():
                #pre processing:
                input = row[key_in][:511]+'</s>' 
                labels = row[key_out][:511]+'</s>'
                
                inputbatch.append(input)
                labelbatch.append(labels)
            if not inputbatch:
                break
            inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,pad_to_max_length=True,return_tensors='pt')["input_ids"]
            labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,pad_to_max_length=True,return_tensors="pt") ["input_ids"]
            inputbatch=inputbatch.to(dev)
            labelbatch=labelbatch.to(dev)

            # clear out the gradients of all Variables 
            optimizer.zero_grad()

            # Forward propogation
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_num=loss.item()
            logits = outputs.logits
            running_loss+=loss_num
            if i%10 ==0:      
                loss_per_10_steps.append(loss_num)
            # out.update(progress(loss_num,i, num_of_batches+1))

            # calculating the gradients
            loss.backward()

            #updating the params
            optimizer.step()

        running_loss=running_loss/int(num_of_batches)
        print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))
        

        
    #show plt
    steps = [i*100 for i in range(len(loss_per_10_steps))]
    plt.plot(steps, loss_per_10_steps)
    plt.title('Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(file_name+'.png')
    
    #save model
    model.save_pretrained(file_name+".h5")
    
    return model



name = 't5_step9_15_9_2021'
model2 = train_model('unsupervised.csv','t5-base',name,'input', 'output', 50)
model2.save_pretrained(name+".h5")
model3 = train_model('aty.csv',name+'.h5',name+"2",'QA', 'Atypical_Annot2', 50)
model3.save_pretrained(name+'2'+".h5")
