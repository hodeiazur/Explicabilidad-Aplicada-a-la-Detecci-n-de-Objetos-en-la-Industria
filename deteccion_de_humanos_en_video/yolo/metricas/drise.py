import torch
from rise import RISE
from tqdm import tqdm
from shapely.geometry import Polygon


class DRISEBatch(RISE):
    def __init__(self, model, input_size, device ='cpu', N=100, p1=0.1, gpu_batch=100):
        super(DRISEBatch, self).__init__(model, input_size, device, N, p1, gpu_batch)
    
    def calculate_iou(self, polygon1, polygon2):
        """
        Calculate Intersection over Union (IoU) of two polygons.
        """
        
        # print('polygon1:',polygon1)
        # print('polygon2:',polygon2)
        poly1 = Polygon(polygon1)
        poly2 = Polygon(polygon2)

        if not poly1.is_valid or not poly2.is_valid:
            print('not valid')
            return 0.0

        # Calculate intersection area 
        inter_area = poly1.intersection(poly2).area
        # Calculate union area
        union_area = poly1.union(poly2).area

        # Compute the IoU
        iou = inter_area / union_area if union_area > 0 else 0

        return iou
    
    def process_in_batches(self, stack):
        """
        Process the masked images in batches and handle variable detection outputs.
        Collects results in a list to accommodate varying numbers of detections.
        """
        results = []  # Initialize an empty list to hold the detection results
        with torch.no_grad():  # Ensure no gradients are computed
            for i in range(0, stack.size(0), self.gpu_batch):
                # Process the batch through the model
                batch_results = self.model(stack[i:i+self.gpu_batch])  
                # Instead of concatenating, append each batch's results to the list
                results.extend(batch_results)
        return results
    
    def apply_masks_in_batches(self, x, masks_chunk):
        """
        Apply the generated masks to the input image in chunks.
        """
        stack = torch.mul(masks_chunk, x)  # Apply masks
        return stack

    def calculate_contributions(self, p, masks_chunk, target_class_index, target_bbox, contributions, aggregation):
        """
        Calculate the contributions for the saliency map based on IoU and class scores.
        """
        
        for i in range(len(p)):
            boxes = p[i]['boxes']
            labels = p[i]['labels']
            scores = p[i]['scores']

            max_contribution = None
            max_points = 0
            mask_contributions = []

            for j in range(len(boxes)):
                score = float(scores[j].item())
                label = int(labels[j].item())

                if label == target_class_index:
                    bbox = boxes[j].cpu().numpy()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bbox
                    vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                    iou_score = self.calculate_iou(polygon1=target_bbox, polygon2=vertices)

                    if iou_score > 0:
                        points = iou_score * score
                        current_contribution = masks_chunk[i, 0, :, :].cpu() * points

                        if max_contribution is None or points > max_points:
                            max_points = points
                            max_contribution = current_contribution

                        mask_contributions.append(current_contribution)

            
            # Add the contribution of the mask
            if aggregation == 'max' and max_contribution is not None:
                contributions.append(max_contribution)
            elif aggregation == 'avg' and mask_contributions:
                avg_contribution = torch.mean(torch.stack(mask_contributions), dim=0)
                contributions.append(avg_contribution)
            elif aggregation == 'sum' and mask_contributions:
                summed_contribution = torch.sum(torch.stack(mask_contributions), dim=0)
                contributions.append(summed_contribution)
                    
        return contributions

    def aggregate_contributions(self, contributions, max_contribution, H, W, aggregation='sum'):
        """
        Aggregate the contributions to form the final saliency map.
        
        Normalizes the output
        """
        saliency_map = torch.zeros((H, W), device='cpu') # contributions increases too much; better to handle at RAM

        # Apply selected aggregation method
        if contributions:
            if aggregation == 'sum':
                for contribution in contributions:
                    saliency_map += contribution
            elif aggregation == 'avg':
                for contribution in contributions:
                    saliency_map += contribution
                saliency_map /= len(contributions)
            # elif aggregation == 'max' and max_contribution is not None:
            #     saliency_map += max_contribution
        
            # ***Normalization
            if saliency_map.max!=0:
                saliency_map /= saliency_map.max()
        
        else:
            print("Warning: No contributions provided for aggregation.")
        
        return saliency_map.cpu().numpy()

    def forward(self, x, target_class_indices, target_bbox):
        """
        Forward pass adapted for object detection to generate saliency maps for multiple classes.
        
        Parameters:
        - x: Input image tensor.
        - target_class_indices: A list of class indices for which to generate saliency maps.
        
        Returns:
        A dictionary of saliency maps, keyed by the target class indices.
        """
    
        _, _, H, W = x.size()
        
        saliency_maps = {}
        contributions = []  # To keep track of all contributions for aggregation
        chunk_size = self.gpu_batch
        
        
        # ***PROCESS EVERYTHING IN CHUCKS***
        for i in tqdm(range(0, self.N, chunk_size)):
            # Split masks
            masks_chunk = self.masks[i:min(i + chunk_size, self.N)].to(self.device)  # Load only the necessary chunk to GPU
            # Apply masks to the image for the current chunk
            stack = self.apply_masks_in_batches(x, masks_chunk)
            p = self.process_in_batches(stack)
                        
            for target_class_index in target_class_indices:
                contributions = self.calculate_contributions(
                    p=p,
                    masks_chunk=masks_chunk, 
                    target_class_index=target_class_index, 
                    target_bbox=target_bbox, 
                    contributions=contributions, 
                    aggregation='max',
                )

            
        # Calculate aggregated & normalized saliency maps
        for target_class_index in target_class_indices:
            saliency_maps[target_class_index] = self.aggregate_contributions(
                contributions=contributions,
                max_contribution=None,
                H=H,W=W,
                aggregation='sum' #sum
            )
        
        return saliency_maps


