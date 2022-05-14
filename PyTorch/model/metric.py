from typing import Callable, Sequence, Union
import torch
from utils.util import ssim as ssim_fn 

#def eval_step(engine, batch):
#    return batch
#default_evaluator = Engine(eval_step)


class MetricEval(object):
    r""" Compute Peak Signal to Noise Ratio metric (PSNR) """

    @staticmethod
    def psnr(gt: Sequence[torch.Tensor],
            y_pred: Sequence[torch.Tensor],
            device: Union[str, torch.device] = torch.device("cpu"),
    )-> torch.Tensor:
        psnr_ = torch.tensor(0.0, dtype=torch.float64, device=device)
        B = gt.shape[0]

        if gt.dtype != y_pred.dtype:
            raise TypeError(
                f"Expected gt and y_pred to have the same data type. Got y_pred: {gt.dtype} and y: {y_pred.dtype}."
            )

        if gt.shape != y_pred.shape:
            raise ValueError(
                f"Expected gt and y_pred to have the same shape. Got y_pred: {gt.shape} and y: {y_pred.shape}."
            )

        if torch.max(gt) <= 1.:
            gt = torch.round(gt * 255.)
            y_pred = torch.round(y_pred * 255.)

        dim = tuple(range(1, gt.ndim))
        mse_error = torch.pow(y_pred.double() - gt.view_as(y_pred).double(), 2).mean(dim=dim)
        psnr_ = torch.sum(20.0 * torch.log10(255. / torch.sqrt(mse_error + 1e-10))).to(device=device)
        return psnr_/ B


    @staticmethod
    def ssim(gt: Sequence[torch.Tensor],
            y_pred: Sequence[torch.Tensor],
            device: Union[str, torch.device] = torch.device("cpu"),
            data_range: Union[int, float]=1.,
    )-> torch.Tensor:
        ssim_ = torch.tensor(0.0, dtype=torch.float64, device=device)
        B = gt.shape[0]

        if gt.dtype != y_pred.dtype:
            raise TypeError(
                f"Expected gt and y_pred to have the same data type. Got y_pred: {gt.dtype} and y: {y_pred.dtype}."
            )

        if gt.shape != y_pred.shape:
            raise ValueError(
                f"Expected gt and y_pred to have the same shape. Got y_pred: {gt.shape} and y: {y_pred.shape}."
            )

        if torch.max(gt) <= 1. or  data_range==1:
            gt = torch.round(gt * 255.)
            y_pred = torch.round(y_pred * 255.)

        #metric = SSIM(data_range=255.)
        #metric.attach(default_evaluator, 'ssim')
        #state = default_evaluator.run([[y_pred, gt]])
        #ssim_ = state.metrics['ssim']
        ssim_ = ssim_fn(gt, y_pred)
        return ssim_
