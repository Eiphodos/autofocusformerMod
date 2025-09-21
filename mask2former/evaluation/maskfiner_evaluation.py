from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
import numpy as np
import random
import torch
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision.transforms.functional import to_pil_image
from collections import OrderedDict
import os
from PIL import Image


class MaskFinerSemSegEvaluator(SemSegEvaluator):

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
            self.save_disagreement_masks(input, output)

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)

            self.save_error_map(pred, gt, input["file_name"])

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

    def save_disagreement_masks(self, inp, outp):
        inference_out_dir = os.path.join(self._output_dir, 'inference_output')
        os.makedirs(inference_out_dir, exist_ok=True)
        fp = inp['file_name']
        fn = os.path.splitext(os.path.basename(fp))[0]

        ss = outp["sem_seg"].argmax(dim=0).to(self._cpu_device)
        ss = np.array(ss, dtype=int)

        hsv_colors = [(i / self._num_classes, 0.75, 0.75) for i in range(self._num_classes)]
        random.Random(1337).shuffle(hsv_colors)
        rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
        color_map = (np.array(rgb_colors) * 255).astype(np.uint8)
        H, W = ss.shape
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        for label in range(self._num_classes):
            rgb_image[ss == label] = color_map[label]
        image = Image.fromarray(rgb_image)

        image.save(os.path.join(inference_out_dir, fn + '_sem_seg.png'))
        np.save(os.path.join(inference_out_dir, fn + '_sem_seg_raw.npy'), ss)

        disagreement_masks_only_dict = {k:v for k, v in outp.items() if "disagreement_mask_" in k}
        for k, v in disagreement_masks_only_dict.items():
            ml_out = outp[k]
            scale = k[-1]
            plt.imsave(os.path.join(inference_out_dir, f"{fn}_{k}.png"), np.asarray(ml_out), cmap='gray', vmin=0, vmax=255)


    def save_input_image(self, image_tensor, fp):
        im = Image.fromarray(image_tensor.detach().cpu().squeeze().permute(1,2,0).numpy(), 'RGB')

        inference_out_dir = os.path.join(self._output_dir, 'inference_output')
        fn = os.path.splitext(os.path.basename(fp))[0]

        im.save(os.path.join(inference_out_dir, fn + '_input_image.png'))

    def save_error_map(self, pred, gt, fp):
        H, W = pred.shape
        error = np.zeros((H, W), dtype=np.uint8)
        empty = np.zeros((H, W, 2), dtype=np.uint8)
        error[pred != gt ] = 255
        error[gt == self._ignore_label] = 0
        error_rgb = np.concatenate([np.expand_dims(error, axis=2), empty], axis=2)
        im = Image.fromarray(error_rgb, 'RGB')

        inference_out_dir = os.path.join(self._output_dir, 'inference_output')
        fn = os.path.splitext(os.path.basename(fp))[0]

        im.save(os.path.join(inference_out_dir, fn + '_error.png'))


class MaskFinerCOCOSemSegEvaluator(SemSegEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir
        )
        meta = MetadataCatalog.get(dataset_name)
        self.data_id_to_cont_id = meta.stuff_dataset_id_to_contiguous_id

        # Fix ignore label and "no class"
        self.data_id_to_cont_id[self._ignore_label] = self._ignore_label

        # Fix removed classes
        # Using Caffe-compatible labels which makes the indexes shifted by -1
        self.data_id_to_cont_id[11] = self._ignore_label #street sign
        self.data_id_to_cont_id[25] = self._ignore_label #hat
        self.data_id_to_cont_id[28] = self._ignore_label #shoe
        self.data_id_to_cont_id[29] = self._ignore_label #eye glasses
        self.data_id_to_cont_id[44] = self._ignore_label #plate
        self.data_id_to_cont_id[65] = self._ignore_label #mirror
        self.data_id_to_cont_id[67] = self._ignore_label #window
        self.data_id_to_cont_id[68] = self._ignore_label #desk
        self.data_id_to_cont_id[70] = self._ignore_label #door
        self.data_id_to_cont_id[82] = self._ignore_label #blender
        self.data_id_to_cont_id[90] = self._ignore_label #hair brush

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
            self.save_disagreement_masks(input, output)

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)

            dataset_classes, inverse = np.unique(gt, return_inverse=True)
            mapped = np.array([self.data_id_to_cont_id[x] for x in dataset_classes]).astype("int")
            gt = mapped[inverse].reshape(gt.shape)


            self.save_error_map(pred, gt, input["file_name"])

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

    def save_disagreement_masks(self, inp, outp):
        inference_out_dir = os.path.join(self._output_dir, 'inference_output')
        os.makedirs(inference_out_dir, exist_ok=True)
        fp = inp['file_name']
        fn = os.path.splitext(os.path.basename(fp))[0]

        ss = outp["sem_seg"].argmax(dim=0).to(self._cpu_device)
        ss = np.array(ss, dtype=int)

        hsv_colors = [(i / self._num_classes, 0.75, 0.75) for i in range(self._num_classes)]
        random.Random(1337).shuffle(hsv_colors)
        rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
        color_map = (np.array(rgb_colors) * 255).astype(np.uint8)
        H, W = ss.shape
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        for label in range(self._num_classes):
            rgb_image[ss == label] = color_map[label]
        image = Image.fromarray(rgb_image)

        image.save(os.path.join(inference_out_dir, fn + '_sem_seg.png'))
        np.save(os.path.join(inference_out_dir, fn + '_sem_seg_raw.npy'), ss)

        disagreement_masks_only_dict = {k:v for k, v in outp.items() if "disagreement_mask_" in k}
        for k, v in disagreement_masks_only_dict.items():
            ml_out = outp[k]
            scale = k[-1]
            plt.imsave(os.path.join(inference_out_dir, f"{fn}_{k}.png"), np.asarray(ml_out), cmap='gray', vmin=0, vmax=255)


    def save_input_image(self, image_tensor, fp):
        im = Image.fromarray(image_tensor.detach().cpu().squeeze().permute(1,2,0).numpy(), 'RGB')

        inference_out_dir = os.path.join(self._output_dir, 'inference_output')
        fn = os.path.splitext(os.path.basename(fp))[0]

        im.save(os.path.join(inference_out_dir, fn + '_input_image.png'))

    def save_error_map(self, pred, gt, fp):
        H, W = pred.shape
        error = np.zeros((H, W), dtype=np.uint8)
        empty = np.zeros((H, W, 2), dtype=np.uint8)
        error[pred != gt ] = 255
        error[gt == self._ignore_label] = 0
        error_rgb = np.concatenate([np.expand_dims(error, axis=2), empty], axis=2)
        im = Image.fromarray(error_rgb, 'RGB')

        inference_out_dir = os.path.join(self._output_dir, 'inference_output')
        fn = os.path.splitext(os.path.basename(fp))[0]

        im.save(os.path.join(inference_out_dir, fn + '_error.png'))


class MaskFinerCityscapesInstanceEvaluator(CityscapesEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def __init__(self, dataset_name, maskfiner=False,  output_dir=None):
        super(MaskFinerCityscapesInstanceEvaluator, self).__init__(dataset_name)
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        meta = MetadataCatalog.get(dataset_name)
        self._num_classes = len(meta.stuff_classes)
        self.maskfiner = maskfiner
        self._inf_dir = os.path.join(self._output_dir, 'inference_instance_output')
        os.makedirs(self._inf_dir, exist_ok=True)


    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            if self.maskfiner:
                self.save_disagreement_masks(input, output)

            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._inf_dir, basename + "_pred.txt")

            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class]
                        class_id = name2label[classes].id
                        score = output.scores[i]
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        png_filename = os.path.join(
                            self._inf_dir, basename + "_{}_{}.png".format(i, classes)
                        )

                        Image.fromarray(mask * 255).save(png_filename)
                        fout.write(
                            "{} {} {}\n".format(os.path.basename(png_filename), class_id, score)
                        )
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval, cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling

        self._logger.info("Evaluating results under {} ...".format(self._inf_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._inf_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._inf_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        self._working_dir.cleanup()
        return ret

    def save_disagreement_masks(self, inp, outp):
        dis_path = os.path.join(self._inf_dir, "disagreement_masks")
        os.makedirs(dis_path, exist_ok=True)
        fp = inp['file_name']
        fn = os.path.splitext(os.path.basename(fp))[0]

        disagreement_masks_only_dict = {k:v for k, v in outp.items() if "disagreement_mask_" in k}
        for k, v in disagreement_masks_only_dict.items():
            ml_out = outp[k]
            plt.imsave(os.path.join(dis_path, fn + '_' + k + '.png'), np.asarray(ml_out), cmap='afmhot')


class MaskFinerCityscapesSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def __init__(self, dataset_name, maskfiner=False, output_dir=None):
        super(MaskFinerCityscapesSemSegEvaluator, self).__init__(dataset_name)
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        meta = MetadataCatalog.get(dataset_name)
        self._num_classes = len(meta.stuff_classes)
        self.maskfiner = maskfiner
        self._inf_dir = os.path.join(self._output_dir, 'inference_output')
        os.makedirs(self._inf_dir, exist_ok=True)

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):

            self.save_disagreement_masks(input, output)

            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy()
            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval, cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        self._working_dir.cleanup()
        return ret

    def save_disagreement_masks(self, inp, outp):

        fp = inp['file_name']
        fn = os.path.splitext(os.path.basename(fp))[0]

        ss = outp["sem_seg"].argmax(dim=0).to(self._cpu_device)
        ss = np.array(ss, dtype=int)

        hsv_colors = [(i / self._num_classes, 0.75, 0.75) for i in range(self._num_classes)]
        random.Random(1337).shuffle(hsv_colors)
        rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
        color_map = (np.array(rgb_colors) * 255).astype(np.uint8)
        H, W = ss.shape
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        for label in range(self._num_classes):
            rgb_image[ss == label] = color_map[label]
        image = Image.fromarray(rgb_image)

        image.save(os.path.join(self._inf_dir, fn + '_sem_seg.png'))
        #np.save(os.path.join(self._inf_dir, fn + '_sem_seg_raw.npy'), ss)

        if self.maskfiner:
            disagreement_masks_only_dict = {k:v for k, v in outp.items() if "disagreement_mask_" in k}
            for k, v in disagreement_masks_only_dict.items():
                ml_out = outp[k]
                plt.imsave(os.path.join(self._inf_dir, fn + '_' + k + '.png'), np.asarray(ml_out), cmap='afmhot')
