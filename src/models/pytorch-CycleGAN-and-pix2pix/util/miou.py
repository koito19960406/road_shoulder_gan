import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class mIoU:
    def __init__(self, opt, model_name="facebook/mask2former-swin-tiny-cityscapes-semantic"):
        """
        Initialize the Segmenter with a model and dataset.

        Args:
            model_name (str): The name of the pre-trained model.
            opt (object): An options object with use_segmentation attribute.
        """
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.use_segmentation = opt.use_segmentation
        if self.use_segmentation:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
            
    def __call__(self, real, fake):
        """
        Calculate negative mIoU between real and fake images based on a condition.

        Args:
            real (torch.Tensor): A tensor of real images with shape (batch_size, channels, height, width).
            fake (torch.Tensor): A tensor of fake images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor or int: The negative mIoU between the real and fake images if use_segmentation is true, otherwise 0.
        """
        if not self.use_segmentation:
            return 0

        segmented_real = self._semantic_segmentation(real)
        segmented_fake = self._semantic_segmentation(fake)

        intersect = torch.logical_and(segmented_real, segmented_fake)
        union = torch.logical_or(segmented_real, segmented_fake)
        IoU = torch.sum(intersect) / torch.sum(union)
        
        return 1 - IoU

    def _semantic_segmentation(self, images):
        """
        Perform semantic segmentation on the given images.

        Args:
            images (torch.Tensor): A tensor of input images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Semantic segmentation output tensor.
        """
        inputs = self.processor(images=images.cpu().numpy(), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        segmentations = self.processor.post_process_semantic_segmentation(outputs, target_sizes=inputs["input_ids"].shape[-2:])
        segmentations_tensor = torch.from_numpy(segmentations).to(self.device)
        
        return segmentations_tensor
