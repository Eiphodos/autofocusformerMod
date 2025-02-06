from detectron2.evaluation import SemSegEvaluator
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import os

class MetaLossSemSegEvaluator(SemSegEvaluator):

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            self.save_sem_seg_metaloss_predictions(input, output)

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def save_sem_seg_metaloss_predictions(self, inp, outp):
        inference_out_dir = os.path.join(self._output_dir, 'inference_output')
        os.makedirs(inference_out_dir, exist_ok=True)
        ss = outp["sem_seg"].argmax(dim=0).to(self._cpu_device)
        ss = np.array(ss, dtype=int)
        fn = inp['file_name']
        ss_pred = to_pil_image(ss)
        plt.imsave(os.path.join(inference_out_dir, fn + '_sem_seg.png'), np.asarray(ss_pred))
        meta_loss_only_dict = {k:v for k, v in outp.items() if "meta_loss_candidates_scale_" in k}
        for k, v in meta_loss_only_dict.items():
            ml_out = to_pil_image(outp[k])
            scale = k[-1]
            plt.imsave(os.path.join(inference_out_dir, fn + 'meta_loss_scale_{}.png'.format(scale)), np.asarray(ml_out))