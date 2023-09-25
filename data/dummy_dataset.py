from torch.utils.data import Dataset
import torch

class DummyDataset(Dataset):
    def __init__(self,image_w_h,d_model,max_source_length,max_target_length,length):
        super.__init__(self)
        self.image_w_h = image_w_h
        self.d_model = d_model
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.length = length
    def __getitem__(self,idx):
        image = torch.rand(3,self.image_w_h[0],self.image_w_h[1])
        src_text = torch.rand(self.max_source_length,self.d_model)
        tgt_text = torch.rand(self.max_target_length,self.d_model)
        
        return image,src_text,tgt_text
    def __len__(self):
        return self.length
