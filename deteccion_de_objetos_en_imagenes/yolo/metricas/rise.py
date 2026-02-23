import torch
from tqdm import tqdm
from base import PerturbationBase

class RISE(PerturbationBase):
    def __init__(self, model, input_size, device='cpu', N=100, p1=0.1, gpu_batch=100):
        super(RISE, self).__init__(input_size, device, N, p1)
        self.model = model
        self.gpu_batch = gpu_batch

    def apply_masks(self, x):
        """
        Apply the generated masks to the input image.
        """
        self.masks = self.masks.to(self.device) # move to gpu
        stack = torch.mul(self.masks, x.data)
        return stack
 
    def calculate_saliency_map(self, p, H, W):
        """
        Calculate the saliency map-s from model predictions.
        """
        N = self.N
        CL = p.size(1)  # Number of classes
        
        # generate a saliency map per class
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        # reshape saliency maps
        sal = sal.view((CL, H, W))
        # normalize
        # sal = sal / N / self.p1 # equivalent to --> (sal) / (N*self.p1)
        sal /= sal.max()
        
        return sal.cpu().numpy()

    def process_in_batches(self, stack):
        """
        Process the masked images in batches to avoid GPU memory overload.
         -In the case of Resnet50 trained on ILSVRC2012, 100 classes have to be predicted. 
         Therefore, the output (p) is expected to have size of [N, 100]
        """
        p = []
        with torch.no_grad(): # very important to not build a graph --> VRAM of nvidia is HUGELY consumed!
            for i in tqdm(range(0, self.N, self.gpu_batch)):
                print('processed {}/{}'.format(i, self.N))
                batch_output = self.model(stack[i:min(i + self.gpu_batch, self.N)])
                p.append(batch_output)
     
        p = torch.cat(p,dim=0)
        return p

    def forward(self, x):
        """
        The forward pass applies masks, processes inputs in batches, and calculates the saliency map.
        """
        _, _, H, W = x.size()
        # Apply masks to the image
        stack = self.apply_masks(x)
        # Process in batches
        p = self.process_in_batches(stack)
        # Calculate saliency map
        saliency_maps = self.calculate_saliency_map(p, H, W)
        
        return saliency_maps
    
    
# class RISEBatch(RISE):
#     def forward(self, x):
#         # Apply array of filters to the image
#         N = self.N
#         B, C, H, W = x.size()
#         stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
#         stack = stack.view(B * N, C, H, W)
#         stack = stack

#         #p = nn.Softmax(dim=1)(model(stack)) in batches
#         p = []
#         for i in range(0, N*B, self.gpu_batch):
#             p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
#         p = torch.cat(p)
#         CL = p.size(1)
#         p = p.view(N, B, CL)
#         sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
#         sal = sal.view(B, CL, H, W)
#         return sal